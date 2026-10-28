# NeuPIMs 源码与论文参考

参考源码 commit：`f299af3fc8f20077e816f2e48313294cddb4bd7c`，位于相邻的 `../NeuPIMs/`。未构建或运行 NeuPIMs；它只用于参考，不用于替代 Pimba。

## 论文依据

已阅读论文的架构、memory controller 与 sub-batch scheduling 相关部分。NeuPIMs 将两个独立子批的 NPU GEMM 与 PIM attention 交错执行，利用专门的双 row buffer 和混合命令调度；因此软件拆 batch 本身不保证硬件可并行。Pimba 使用 GPU 与 SPU/MX8，不能默认拥有 NeuPIMs 的硬件能力。[NeuPIMs 论文 §5–6](https://arxiv.org/html/2403.00579v1)

Pimba 论文描述的 SPU 在两个 bank 之间进行访问交错，目标是状态更新的效率；这不是不同 sub-batch 的 GPU/PIM 并行执行。[Pimba 论文 §4–6](https://arxiv.org/html/2507.10178v1)

## 实际源码机制

| 研究问题 | 源码证据 | 对 Pimba 的含义 |
|---|---|---|
| 如何分 batch | `src/scheduler/Scheduler.cc:287` `group_sub_batches` | sub_batch_mode 关闭时合并到 _breq1；开启时逐 channel 分成 _breq1/_breq2；simple 分半并轮换奇数归属，另一分支按估计 latency 使用 partition_lists_dp |
| latency 估计 | `Scheduler.cc:267` `estimate_mha_latency` | 依据 input_size、页大小、banks、heads、gemv/global-write latency 估计 attention 负载，不适合直接衡量 Mamba 固定状态尺寸的 SU |
| GPU/NPU vs PIM 排序 | `Scheduler.cc:245` 生成两个 StageProgram；`StageProgram.cc:28` 的 A–F 表 | stage 奇偶交换两批的 SA/PIM 身份；A/F 填充与排空，C/D 是带 FFN 的 transformer 流水，不能照搬到 Mamba |
| stage 内依赖 | `StageProgram.cc:164` `find_executable_node`，`:182` `finish_operation` | tensor child operations、check_executable；前驱完成后释放 ready operations |
| 同步 | `Scheduler.cc:596` `refresh_stage` | 只有两个 program 都为 null 才推进 stage；不是任意无屏障重叠 |
| completion | `Scheduler.cc:517` `finish_tile` | remain_tiles 减到0后 finish_operation，再 refresh_status1/2 |
| 独立 request 状态 | `Scheduler.cc:561` `cleanup_sub_batch` | 更新每个 InferRequest 的 generated/is_initiated，达到 output_size 后放入 completed queue；有各自 K/V cache |
| 共同模拟时间 | `Simulator.cc:115–245` | CORE/DRAM/ICNT cycle mask 驱动，分别发 SA/PIM tiles，收到 completion，实际时间由 tick 和资源服务决定 |
| memory contention | `Simulator.cc:187–237` | 两路 memory_request/ICNT queues 在同一 _dram 上检查 is_full 并 push；不能把两路都当作无竞争无限带宽 |
| README 范围 | `README.md` Baselines | 文档说明 interleaving 在 batch>=256 启用；源码开关是 sub_batch_mode。没有把该阈值当成 Pimba 的要求 |

A–F 的源码表：SA 依次为 QKV1、QKV2、Proj/FFN/QKV1、Proj/FFN/QKV2、Proj/FFN1、Proj/FFN2；PIM 依次为空、MHA1、MHA2、MHA1、MHA2、空。依赖由 stage 屏障与 stage 内 executable nodes 共同维护。不能单独取每阶段 max(time1,time2) 代替整个 simulator 的资源行为。

## 适用与不适用部分

可以借鉴独立 sub-batch 身份、ready/completion 机制、保持前驱约束、填充/排空、按服务需求而非仅 request 数分组。Pimba 初版可支持多个等长 sub-batch，而无需动态 arrival 或实际 PyTorch requests。

不能复制 SA tile 计算模型、transformer FFN stage 表、KV 长度估计器或双 row buffer。Pimba 应保留原 GPU tile/SM 利用率模型、PIM state-update trace、SPU/MX8 和 command readiness。Mamba 必须增加每 sub-batch/decoder 的跨 token state version 顺序，不能只保留单次 GPU→PIM 边。

小 sub-batch 可能降低 GPU tile/SM 利用率、增加重复权重读取、同步与填充排空，也可能损失 PIM channel 填充；越小不一定越快。GPU 是 roofline 解析模型而 NeuPIMs 是 SA tile/cycle 驱动，这个抽象层差异必须保留在结果可信度说明中。
