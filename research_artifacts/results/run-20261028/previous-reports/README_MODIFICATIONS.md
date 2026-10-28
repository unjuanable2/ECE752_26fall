# 本次工作与限制

任务尚未完成：原始 Pimba 构建依赖被环境阻塞，未获取 baseline，未修改模拟器执行路径，也没有性能实验结果。

Pimba 源码已在工作目录可用，commit `0a902eeafddbfddaa34723ed74a761e4453b870d`；保留原仓库历史并已创建本地分支 `feature/sub-batch-interleaving`。NeuPIMs 仅作参考，未改动，未 push。

新增文件：

- `BATCH_EXECUTION_MODEL.md`：包含实际 Python/C++ 文件、函数和依赖/并发边界的源码分析。
- `NEUPIMS_REFERENCE.md`：两组 sub-batch、stage、completion、依赖和 memory traffic 的源码对照。
- `IMPLEMENTATION.md`：明确标为未实施的设计与验证约束。
- `EXPERIMENTS.md`、`RESULTS.md`：真实失败命令、恢复步骤和缺失结果说明。
- `scripts/research_preflight.py`：仅用标准库检查本地 Python、工具、包版本与 simulator binary。
- `results/`：环境诊断与构建/smoke 失败日志；没有模拟测量数据。
- `figures/STATUS.md`：无数据，未生成性能图。

未修改 `src/`、`csrc/`、上游硬件配置、SPU、MX8、memory timing 或 GPU 模型。

诊断脚本已用两个现有 Python 环境执行，均正确报告 prerequisites 不满足（exit1）。这不是 simulator 功能测试或 interleaving 验证。
