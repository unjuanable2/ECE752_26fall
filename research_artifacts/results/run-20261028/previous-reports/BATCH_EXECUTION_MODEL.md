# 原始 Pimba 的 batch/request 执行模型

审计对象：Pimba commit `0a902eeafddbfddaa34723ed74a761e4453b870d`。本文件基于实际源码；原始构建与 smoke test 因依赖缺失尚未成功，所以这些结论属于源码分析，不是运行测量。用户要求先成功 baseline 再修改执行机制，当前未改执行路径。构建受阻期间仅推进可独立完成的只读分析。

## 结论

**性能模拟路径不显式创建每个 inference request。** Batch=128 进入算子矩阵形状或算子数量，不会创建 128 个有独立状态、依赖和完成时间的 request。Mamba-2 模型也没有按每个 token/decoder 展开完整执行图，而是把同类算子的重复次数折叠到 `num_layers`，最后对其 latency 相乘、求和。因此现有 workload 不能直接用于独立 sub-batch 的 event scheduling，必须先扩展表示。

注意区分 `src/pimba/accuracy/` 的真实张量/推理实现与 `src/pimba/performance/` 的性能模拟。前者的状态张量不是后者的 request scheduler。

## 实际模块对应

| 功能 | 文件、类、函数 | 实际行为 |
|---|---|---|
| 用户入口、batch/序列参数 | `src/pimba/performance/exp.py:9` `Performance.__init__`, `setup`, `run` | 创建一个 model 和 system；API 默认 lin=lout=2048 |
| 配置到模型 | `performance/models/utils.py` `create_model`；`models/models.yaml` | model 名称解析为 architecture/preset；batch、lin、lout、tp、state_dbyte 传入模型 |
| Workload 容器 | `performance/models/base.py:6` `Model` | 只有 `self.layers: list[Layer]`，迭代返回算子列表 |
| Workload 算子 | `performance/layers/base.py:4` `Layer`, `FC`, `ACT`, `NORM`, `SU` | 保存 phase/category/name/m/n/k/num_op/dbyte/num_layers；没有 request_id、时间戳或依赖边 |
| Mamba-2 | `performance/models/mamba2.py:7` `Mamba2.__init__` | 构造聚合 decode 算子模板 |
| 算子映射、系统执行 | `performance/systems/system.py:36` `mapping`, `simulate` | gen 的 state_update/gemv 映射 PIM，其余 GPU；逐模板求 latency 并乘重复次数 |
| GPU | `performance/devices/gpu.py` `GPU.get_optimal_tile`, `compute_time`, `GPU_FC.time` 等 | tile/cache/线程块利用率与带宽、算力解析模型；没有 GPU 指令或 kernel completion events |
| Handler dispatch | `performance/devices/base.py` `Device.simulate`, `Handler.simulate` | 按 Layer 类型同步计算 time、energy |
| PIM | `performance/devices/pim.py:57` `run_ramulator2`, `PIM_SU.time/power` | 生成单个聚合算子 trace，同步调用 C++ executable，返回 cycles 和能耗 |
| PIM trace | `performance/utils/trace.py:15` `Trace.generate` | 按状态尺寸、num_op、channels 构造 chunk、reg write、COMP、result、refresh barrier |
| PIM 配置 | `performance/systems/utils.py` `create_system`；`systems/{gpu,pim}.yaml` | A100 的 5 stacks ×8 =40 channels，1512 MHz；Pimba state_dbyte=1，保留原命令调度与 chunk group |
| C++ 时间推进 | `csrc/main.cpp:21` `main` | tick frontend 和 memory_system，直到 trace 全完成；只推进此 PIM trace 的时间 |
| DRAM controller | `csrc/controllers/lax_controller.cpp` `LAXController.tick`, `schedule_request`, `serve_completed_reads` | 有多个未完成 memory requests、buffers、FRFCFS readiness 与 callback |
| DRAM 时序 | `csrc/dram_specs/LAX.cpp` `LAX.check_ready`, `issue_command`, `set_timing_vals` | DRAM readiness、PIM 命令顺序、refresh barrier 和 timing constraints |
| 地址映射 | `csrc/addr_mappers/lax_mapper.cpp` `LAXMapper` | 对 PIM trace 编码的 channel/pseudochannel/ordering 元信息映射 |
| 统计 | `systems/system.py:46`, `exp.py:73`, `csrc/main.cpp` | 每模板 time/能耗，按 category 求和；C++ 导出 cycles/num_act/num_comp/num_rdwr |

表中省略的 `performance/` 路径前缀均为 `src/pimba/performance/`。

## Request representation 与 batch 影响

`Model.__init__` 只有 layers 列表；`Layer.__init__` 保存纯整数维度等元信息。`Mamba2.__init__` 并无按 batch 遍历 request 的循环。单 GPU 的模板顺序是：

```text
qkv -> causal_conv1d -> discretization -> state_update -> norm -> act -> o
 GPU       GPU              GPU              PIM          GPU     GPU   GPU
```

以 mamba2-2.7b、tp=1、batch=128 为例，源码配置为 decoders=64、hidden_dim=2560、heads=80、head_dim=128、state_dim=64、dbyte=2，Pimba state_dbyte=1。由源码可直接推导：

| 模板 | m | n | k | num_op |
|---|---:|---:|---:|---:|
| qkv/FC |128|10496|2560|1|
| causal_conv1d/ACT |128|21504|1|1|
| discretization/ACT |128|80|1|1|
| state_update/SU |80|128|64|128|
| norm/NORM |128|5120|1|1|
| act/ACT |128|5120|1|1|
| o/FC |128|2560|5120|1|

这是**形状推导，不是性能测量**。tp>1 时还会添加 G2G。每个模板的 `num_layers=64*(lout-1)`。

GPU 的 FC FLOPs 为 `2*m*n*k*num_op`，batch 改变 m，进而改变最优 tile、线程块数、SM 利用率和 HBM traffic。`GPU_FC.time` 取当前算子的 compute_time 与 memory_time 的较大者，这是算子 roofline，不能移用为 GPU/PIM 并行调度。

`PIM_SU.time` 构造 `LayerInfo(hw, "SU", layer.num_op*layer.m, layer.n, layer.k, layer.dbyte)`，因此此例传入 128*80 个 head-state operations。Trace 按 chunks 分配到 channels 和 banks；channel 量化、命令和刷新时序会导致非线性。必须为每个有效 sub-batch size 重新生成形状和 PIM trace，不能把 batch128 的 time 除以4。

不同 request 在性能模型中没有独立数值 state；C++ 的 `Request` 是 DRAM 命令请求，不是用户 inference request。`LAXTrace` 的 address 还承载 mode/order/comps 等元信息，不能当作完整物理 state tensor 的真实地址追踪。

## Sequence length 与执行图

Mamba2 接收 lin 但不在构造体中使用；没有 prefill 算子。lout 通过 `lout-1` 折叠 decode 次数。不能把这里的耗时声称为含 prefill/首 token/排队/采样的端到端请求 latency。lout=1 导致零重复、无 decode 工作量，不是合理性能点。OPT、Zamba2 则存在 `for step in range(1,lout)`，attention 维度使用 lin+step，不能将 Mamba 的压缩方法原样用于这些模型。

原始性能路径没有 per-request DAG，也没有完整 batch DAG 或独立的 GPU/PIM dependency graph。只有体现算法顺序的模板列表。`System.simulate` 对每个模板调用同步 handler，再乘 `num_layers`；这甚至不是把每个 decoder/token 依次执行的真实 trace。其累计时间等效于所有模板重复执行的串行总时长，但没有呈现真实的时间线。

独立 sub-batch 的逻辑依赖应至少展开为 `(sub_batch_id, token_id, decoder_id, op_index)`：同一 decoder 的前后算子链、decoder 输出到下一个 decoder、整个 token 完成到下个 token，以及 state-update 在同一 sub-batch/decoder 上的 token 递归顺序。其他 sub-batch 才可以填补资源空隙。

## 并发能力的边界

| 能力 | 原始系统性能层 | 单次 C++ PIM trace 内 |
|---|---|---|
| GPU/PIM 同时推进共同时间 | 无 | 没有 GPU 模型输入 |
| 多个未完成 GPU/PIM 算子 | 无 | 有多个 memory requests，并非系统算子 |
| Event-based 系统 scheduler | 无 | tick 驱动 DRAM 模型，有请求 callback |
| 显式算子依赖跟踪 | 无 | 有 PIM 命令排序/readiness |
| GPU/PIM 算子 completion events | 无 | 有 trace request completion counter |

Python `subprocess.run` 阻塞等待 PIM trace；其他 GPU 算子的 HBM 读写没有注入同一次 Ramulator execution。因此现有 baseline 没有 GPU/PIM 同时访存时的共享 contention。源码中含 RD/WR 命令定义不等于现有前端已经模拟混合 GPU/PIM traffic：`LAXTrace.init_trace` 接受的是列出的 PIM trace 命令，并不接受普通 read/write trace 文本。

## 需要修改的范围（尚未实施）

1. `performance/exp.py`：增加 interleaving/sub-batch/synchronization 配置，记录 lin/lout 和新的 latency 指标。默认关闭并保持原 `run` 输出兼容。
2. `performance/models/mamba2.py`、`models/base.py`：生成真正按 sub-batch 尺寸构建的模板并展开逻辑执行；不要用聚合 `num_layers` 直接跨依赖调度。
3. 新建 `performance/schedulers/interleaving.py`：ready queues、completion event heap、GPU/PIM resource occupancy、deterministic tie-break 和检查器。
4. `performance/systems/system.py`：默认沿用原路径；启用后调用新 scheduler，不能再把 overlapped operator times 相加当 makespan。
5. `performance/devices/gpu.py`：导出与原 tile 选择一致的 HBM traffic、计算/访存服务需求。保留 GPU 计算模型，不能替换成 NPU。
6. `performance/devices/pim.py`、`utils/trace.py`：按 sub-batch 重新求 PIM duration/commands；若资源配置会变化，cache key 要包含 channels、频率、调度配置，现有 LayerInfo 不含这些字段，不能跨异构配置共用缓存。
7. 若实现共享 HBM 混合请求的时序精度：扩展 `csrc/main.cpp`、`traces/lax_trace.cpp` 和必要的 mapper/controller 接口来接收带就绪时间的 GPU RD/WR，与 PIM 共用 controller。这需要进一步验证地址编码/trace frontend，不是更改一个 Python max 就能完成。
8. 新增 tests、timeline 导出与实验 driver；覆盖非整除尾 batch、1 个 sub-batch、lout边界、所有 token/decoder/state 边、资源互斥和事件时间单调性。

拟增加的数据结构：SubBatch(id, request_range, size, next_token, completion_time)、Operation(id, template, sub_batch, token, decoder, dependencies, remaining_predecessors, resource, duration)、CompletionEvent(time, op_id)、ResourceState(busy_until/queue/channel ownership)、TimelineRecord、每个 sub-batch/decoder 的 StateVersion。它们都是待实现设计，不是现有源码已有对象。

资源限制：共享 HBM channel/bank、CA/DQ bus、refresh、PIM register 写与结果读、GPU weights/activations、重复权重读取和 cache reuse、sub-batch 边界同步/launch/切换。NeuPIMs dual-row-buffer 不能被默认当作 Pimba 已有机制。保留 Pimba SPU/MX8/dataflow 仍需对 GPU/PIM 的同 bank 重叠作约束；近似模型必须显式标注。
