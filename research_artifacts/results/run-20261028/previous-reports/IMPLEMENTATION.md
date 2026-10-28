# 实现状态与源码约束下的设计

**当前没有实现 Sub-batch Interleaving。** 原始 baseline 未能成功运行，按用户“必须先成功运行原始 Pimba，再修改执行机制”的要求，保留 `src/` 和 `csrc/` 原样。下述为设计约束，未作为完成实现或测试结果提交。

已完成源码层面的必要前置分析，见 BATCH_EXECUTION_MODEL.md 与 NEUPIMS_REFERENCE.md。唯一新增 Python 工具是环境诊断 `scripts/research_preflight.py`；不运行性能模型、不改变 timing，不是替代模拟器。

## 最小执行表示

将 Mamba2 原模板按实际 sub-batch size 重建；维护尾 sub-batch 的真实大小和 request 范围。每个 operation 标识为 `(sub_batch, token, decoder, op_index)`。所有前驱 completion 后才能进入对应资源队列。跨 decoder 和跨 token 依赖必须显式表达，SU 的 state 必须按同 sub-batch、同 decoder 的 token 顺序更新。保守依赖链可先按原列表序列连接，避免错误放宽。

不能把 `num_layers` 次重复作为一个长 operation 直接分配给某个 sub-batch，因为这会将原先被压缩的真实算法顺序丢失。可以按需惰性创建下个 decoder/token 的节点，避免 lout=2048 时产生过大对象集合。

scheduler 需要两个 ready queues、completion heap、运行状态和 deterministic tie-break。time 单位统一为 seconds 或整数 ticks；duration 必须正、event 不能倒退。每个 GPU/PIM 同一时刻只服务约定数量的 operation；其他 sub-batch 只在其自身依赖满足时才可 overlap。插入启动、跨设备同步与资源切换 overhead；值需要明确出处和灵敏度分析，不能凭空称硬件实测。

## 共享 HBM 是实现的实质约束

仅同时启动两个独立 Ramulator 进程不能表达共享 HBM。现有 `run_ramulator2` 是孤立 PIM trace，GPU traffic 只用于解析 memory_time；直接把这些 duration 塞入两个资源的事件队列，会得到无 contention 的理想上界，不能作为完整实现交付。

有两条可研究的实现路线：

1. **共同 memory timing timeline**：扩展 frontend 接收就绪时间、request source 和真实 GPU RD/WR，统一经过 mapper/controller/DRAM。必须处理目前 PIM 地址字段承载 ordering/comps 的特殊编码，GPU 地址映射不能覆盖这些字段；再检查 PIM mode 和同 bank 的冲突。GPU 完成要等待其 traffic 与计算服务，而不是提前释放后继。保留 LAX 的原 PIM command timing，不默认引入双 row buffer。
2. **保守资源分区估计**：同一硬件 channel 总数固定，互斥分配 GPU/PIM channel；GPU 带宽和 PIM trace channel 数随可用份额重算。需要显式描述权重、state/activation 放置、容量、数据交接和同步，验证分区实际可实现。缺少这些信息时仍只能作为受限估计，不能宣称 cycle-accurate contention。

目前没有实现或运行任一路线。首先恢复 baseline，随后选择与实际 Pimba 硬件能力一致的机制；不隐含引入新硬件。

## 接口和兼容性

拟在 `Performance.setup`/system 中添加 `sub_batch_interleaving=False`、`sub_batch_size`；默认路径完全保持现有 category time/energy 返回。启用后另外返回 makespan、sub-batch 完成时间和 timeline。不能继续对 overlapped operator time 求和充当 latency。

`LayerInfo` cache key 目前未包含 channel/frequency/config；新调度若允许这些参数变化，应扩大 key 或使用独立 immutable device instance。跨实验保持 upstream 的每次实验独立进程约束，防止缓存污染。

## 待验证项目

- default-off 与不可变 baseline 的逐 category 一致性；单 sub-batch 行为。
- 非整除尾 batch、size>batch/size<=0/lout<=1 的定义和拒绝策略。
- 节点完成计数等于预期，无重复，无遗漏；request 范围恰好覆盖 batch。
- 每条依赖 finish<=successor.start，所有 sub-batch 最终完成，state version 单调。
- GPU/PIM resource 互斥及 channel/bank/总带宽约束；refresh/PIM readiness。
- CPU 实验 wall time 与模拟 latency 分开，makespan 与 operator service-time sum 分开。
- 同步开销及 contention 的敏感性；不能只测理想 overlap。

这些均未执行，不能称 modified simulator 通过 regression tests。
