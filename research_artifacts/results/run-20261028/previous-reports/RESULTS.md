# 当前结果与未完成工作

**原始 baseline 未成功运行；尚无 throughput/latency 性能结果。** 本文件不会把推导形状、论文数据或失败进程耗时记作 simulator 测量。

| 用户要求 | 实际状态 |
|---|---|
| 原始 Pimba 是否成功运行 | 否；构建命令缺 uv/cmake，smoke import 缺 lm_eval，未进入模拟 |
| batch/request 表示 | 已完成源码分析：聚合算子列表，没有独立 request 执行状态；详见 BATCH_EXECUTION_MODEL.md |
| 修改了哪些源代码 | src/、csrc/ 完全未改；仅新增环境诊断脚本及报告 |
| 是否真正实现 interleaving | 否；遵守 baseline 成功后才改执行路径的前置要求 |
| modified 是否通过测试 | 未实现、未测试；preflight不代表正确性测试 |
| baseline throughput/latency | N/A，未测量 |
| modified throughput/latency | N/A，未测量 |
| speedup/latency reduction | N/A，无法计算 |
| 结果可信度 | 构建失败与源码分析有日志/代码依据；不存在可以评估的性能结果 |
| 未完成项 | baseline构建/运行、scheduler实现、共享资源建模、regression、矩阵实验、图表、性能归因 |

## 性能比较表

尚无有效测量行，CSV仅包含 schema；缺失值不是0。

| Model | Batch Size | Sub-batch Size | Original Throughput | Modified Throughput | Speedup | Original Latency | Modified Latency | Latency Reduction |
|---|---|---|---|---|---|---|---|---|

不能判断 throughput 是否提升、latency 是否降低、哪个 size 最好、效果是否随 batch 改变、GPU/PIM idle 是否减少、是否有 degradation、或瓶颈是否移动。

## 已确认的建模限制

- 原始 Mamba2 性能路径无 prefill，lin未使用；仅有折叠的 decode steps。不能声称给出了完整请求端到端 latency。
- GPU 是 tile/roofline 解析模型，PIM 子系统才由 C++ memory timing 模型推进；整个系统不能泛称 cycle-accurate GPU/PIM co-simulation。
- 原始 Python 路径同步运行孤立 PIM trace，没有 GPU 的普通 memory traffic。现有功能不足以验证共享 HBM overlap。
- 原始 PIM trace 的地址还用于传递命令元数据，尚不具备每 request/state 的真实物理地址表示。新并发层必须处理共享资源，不能默认 NeuPIMs 的 dual-row-buffer 已存在。
- 小 sub-batch 会改变 GPU tiling/利用率和 PIM channel/chunk 量化；不能用原 latency除以size个数推算。

adaptive scheduling 值得作为后续方向，但目前只有设计动机，没有实测收益证据。应先验证固定size和共享资源，再根据ready queue、GPU服务需求与PIM等待时间调整size；所有改变都需重新计算真实形状与服务需求。

## 可恢复的交付物

results/status.json 标记当前状态；results/preflight-*.json、build_attempts.json 和逐命令日志是本轮实际记录。原有 clone失败日志仍保存在工作区顶层 results/clone.log，不覆盖。独立分支已创建、未 push。恢复步骤见 EXPERIMENTS.md。
