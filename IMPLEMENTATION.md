# Pimba Sub-batch Interleaving 实现

## 已实现的范围

在原 Pimba `0a902eeafddbfddaa34723ed74a761e4453b870d` 上增加了可运行的 **静态 HBM channel 分区、事件驱动的系统级原型**。先成功编译和运行原始 baseline，保存不可覆盖的结果，然后才修改性能执行路径。Pimba 子仓库分支为 `feature/sub-batch-interleaving`。没有替换 GPU 为 NPU，没有修改 `csrc/`、SPU、MX8、LAX timing 或 PIM dataflow。

此实现真实调度独立 sub-batch 的 operations，但**不是 GPU/PIM 混合访存的联合 cycle-accurate simulator**。PIM duration 来自原 C++ command-timing simulator 的实际执行；系统 makespan 来自新增 Python event simulator。所有 modified 数据必须带此范围说明。

## Workload 与依赖

`src/pimba/performance/schedulers/interleaving.py` 为每个有效 sub-batch size 调用原 `create_model`，重新计算 GPU tile/SM 利用率、访存量和 PIM trace。不是将原始 latency 除以份数。

逻辑 operation ID 为 `(sub_batch, token, decoder, op_index)`。每个 sub-batch 按以下顺序执行：

```
qkv -> causal_conv1d -> discretization -> handoff_in
    -> state_update -> handoff_out -> norm -> act -> o
```

每个 decoder 的最后 operation 连接下一 decoder；最后 decoder 连接下一个 token。每个 sub-batch 都要执行 `64*(lout-1)*9` 个事件节点。这个保守链保证每个 decoder 的 state update 按 token 递增；维护 `state_versions[sub_batch][decoder]`，在 dispatch 和最终 drain 时检查。没有数值张量计算，也没有声称通过 logits/accuracy 验证。

采用惰性展开：每个 sub-batch 同时只释放一个已满足依赖的节点，不必将 2047 个 token 的 DAG 全部保存在内存。GPU/PIM 各自 FIFO ready heap；全局 completion heap 以模拟结束时间推进，处理同一时刻的所有 completion 后释放后继。资源占用严格互斥。其他 sub-batch 可在另一设备执行。记录 request index 范围、completion、operation count、busy/idle/overlap。

## HBM 资源与数据交接

A100/Pimba 总资源保持 40 channels，默认 GPU 20、PIM 20，不能把两边都当成独占 40 channels。逻辑 GPU channel range `[0,20)`，PIM `[20,40)`。C++ 局部 channel ID 映射到对应 PIM 分区；各 PIM trace 仍以该分区的 channels 重新运行。

GPU 的 `peak_memory_bandwidth` 按 channel 份额缩放，其余 GPU 参数不变，继续使用原 Handler。PIM state 全部驻留 PIM 分区，GPU weights/activations 驻留 GPU 分区。同一物理 channel 不同时承担普通 GPU 与 PIM traffic。这样规避同 bank 的 row-buffer 冲突，代价是固定分区的容量和带宽损失，未使用空闲方资源的动态借用。

handoff_in/out 分别计入 GPU 分区的 staging 读/写流量及配置的同步开销，并占用 GPU 资源。PIM 分区的 REG_WRITE/RESULT_READ 已由原 PIM trace 计时，不再重复计入 PIM 时间。每 request/head 的 input staging 为 `(3*state_dim+head_dim)*state_dbyte` bytes；output staging 为 `head_dim*2` bytes，保守按每 head 的完整结果计。它们是显式的系统建模假设，尚未验证实际 Mamba grouped operand reuse、host load queue 或片上转发实现。

默认每次 handoff `sync_ns=1000`，是可配置假设，不是硬件实测。另实际运行 0/5000ns 敏感性实验。`execution_overlap=False` 保持相同分区、形状和交接，只串行 dispatch，可隔离调度 overlap 本身的收益。

原 trace 要求填满 bank wave。尾 sub-batch 不整除时，仅将其 SU dummy slots 填充至 `channels*32 / gcd(channels*32, heads*ceil(head_dim/(32/state_dbyte)))` 的整数倍，并计入真实 padded trace 时间。dummy 不算输出 tokens 或真实 requests。GPU 算子保持实际尾尺寸；例如 batch35/sub16 -> 16,16,3，最后 SU pad 到4。新增 logical state bytes 统计包含 dummy slots，但不含 MX exponent metadata。

## API 与兼容性

```python
from pimba import Performance
with Performance.setup(model='mamba2-2.7b', system='Pimba', num_gpus=1,
        batch_size=128, gpu='A100', lin=2048, lout=2048,
        sub_batch_interleaving=True, sub_batch_size=32,
        pim_channels=20, sync_ns=1000) as exp:
    result = exp.run()
    print(result['timing']['throughput_tokens_s'])
```

开关默认 False，沿用原 `System.simulate` 与原 category 统计输出。True 时返回独立的 `timing` 结果，不对 overlapped service times 求和充当 latency。当前只支持单 GPU 的 Pimba/Mamba2；RetNet/Zamba2 和 TP>1 会明确拒绝。

顶层 `pimba/__init__.py` 改成 lazy import，使纯 CPU performance 路径不再强制导入 accuracy CUDA 依赖。实际请求 Accuracy 时保留上游 Triton deterministic patch。accuracy 流程本次未运行。

`devices/pim.py` 增加 trace flush 与可选 `PIMBA_AUDIT_DIR`，保存 trace/config/stdout/stderr/counters。分区设备通过独立 dataclass instance 和空 cache 创建；不会将不同 channel 配置的 duration 混用。

## 正确性验证与局限

`tests/test_interleaving.py` 包含 baseline 逐 category 精确相等、资源互斥、全节点唯一和完整、所有依赖、独立区间交集验证 overlap、单 sub-batch 无 overlap、尾 batch、跨 token state、配置拒绝、同步敏感性。每个生产实验还执行 runtime invariants。原始和 modified 的归档 PIM trace 已独立 replay，全部 exported counters 精确相等。

限制：没有注入 GPU RD/WR 到同一个 C++ timeline；没有跨 channel 功率/thermal 限制、全局 interconnect 排队、物理地址分配和容量 allocator、MX metadata 容量、片上缓存跨算子复用或异步命令提交器的硬件验证。PIM traces 各自冷启动，refresh phase/controller 状态不跨 SU 持续，沿用原隔离调用抽象。因此结果用于评估此明确映射假设下的趋势，不是原 Pimba 硬件可直接部署的速度预测。系统级 prototype 已实现；高保真联合 memory co-simulation 尚未完成。
