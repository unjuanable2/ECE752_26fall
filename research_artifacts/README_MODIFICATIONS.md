# 修改总结

已完成原始Pimba构建、baseline、batch/request源码分析、NeuPIMs参考分析、可运行的sub-batch事件调度原型、正确性测试、完整Mamba2矩阵实验及图表。没有向远程push。

**实测没有超过原始Pimba。** B128/S64：14923.93→5534.54 tokens/s；decode batch completion latency 17556.77→47341.92ms。相同20/20分区的B128/S32开启overlap则比串行对照快6.75%，但仍显著慢于原始大batch。

## 源码变更

- `pimba/src/pimba/performance/schedulers/interleaving.py`：新增按真实sub-batch尺寸生成profile、独立token/decoder/operator依赖、completion events、state版本、GPU/PIM互斥、channel分区、handoff、尾batch padding与timeline/统计。
- `pimba/src/pimba/performance/schedulers/__init__.py`：新调度API。
- `pimba/src/pimba/performance/exp.py`：默认关闭的interleaving开关、sub_batch_size、pim_channels、sync_ns；默认路径兼容，启用后输出makespan而非服务时间之和。
- `pimba/src/pimba/performance/devices/pim.py`：trace flush和可选原始PIM审计文件导出。
- `pimba/src/pimba/__init__.py`：lazy imports，使CPU performance不再依赖accuracy CUDA包；请求Accuracy时保留Triton patch。
- `pimba/tests/test_interleaving.py`：6项实际执行的回归测试。
- `pimba/scripts/research/`：原始baseline harness、单case/matrix驱动、PIM replay、CSV/绘图脚本及精确依赖lock。

原GPU Handler、Mamba2算子形状定义、PIM trace生成、C++ `csrc/`、SPU、MX8、DRAM timing未修改。总channels仍40，默认静态分配GPU20/PIM20。

## 交付文件

根目录 `BATCH_EXECUTION_MODEL.md`、`NEUPIMS_REFERENCE.md`、`IMPLEMENTATION.md`、`EXPERIMENTS.md`、`RESULTS.md`；`results/run-20261028/`保存原始与完整2048长度实验，`figures/`保存6种图及PDF。原有失败报告归档于`results/run-20261028/previous-reports/`，旧日志保留。

开发分支位于保留上游历史的 `pimba/` 子仓库：`feature/sub-batch-interleaving`。外层ECE752仓库 `.git` 为只读，没有尝试绕过其权限，也没有push任何研究内容。

## 范围

这是具有真实事件/依赖与实际PIM trace timing的**静态channel分区系统级prototype**。共享channel混合GPU/PIM联合memory timing、真实物理地址/容量allocator和accuracy验证未实现。不能将结果描述为新硬件的cycle-accurate实测或保证可以直接部署。具体假设、未完成项与复现命令见IMPLEMENTATION.md和EXPERIMENTS.md。
