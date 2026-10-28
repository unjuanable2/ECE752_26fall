# 实验执行记录与恢复

## 源码和环境

Pimba：`0a902eeafddbfddaa34723ed74a761e4453b870d`，本地分支 `feature/sub-batch-interleaving`。NeuPIMs：`f299af3fc8f20077e816f2e48313294cddb4bd7c`。两者均有原始 Git 历史。未 push。

前一次任务因 clone DNS 失败而停止；本次开始时本地已出现两个仓库，因此继续检查构建。仍有 `git ls-remote ... HEAD` 报 `Could not resolve host: github.com`；不能下载新的依赖，未尝试权限升级、代理、额外凭据、付费服务或绕过网络限制。

## 实际运行的诊断

在本仓库目录：

```bash
python3 scripts/research_preflight.py --output results/preflight-system.json
/home/jianing/pyenv/bin/python scripts/research_preflight.py --output results/preflight-pyenv.json
```

两者均 exit1。Python 都为3.10.12，上游要求 >=3.12,<3.13；uv/cmake/ninja 不在 PATH；没有 build/PimbaSim。两个环境都缺 rich、ruamel.yaml、pandas、lm-eval；pyenv 只有旧 torch2.1.2+cu118、triton2.1.0。依赖版本证据见 JSON。

## 原始构建与 smoke 尝试

在本仓库运行：

```bash
uv sync
cmake --preset release
PYTHONPATH=src /home/jianing/pyenv/bin/python -c 'from pimba import Performance; exp = Performance("mamba2-2.7b", "Pimba", 1, 1, "A100", 16, 2); print(exp.run())'
```

前两个命令无法启动，因为 uv/cmake 不存在。smoke 在导入阶段 exit1：`src/pimba/__init__.py` 急切导入 accuracy，随后缺少 `lm_eval`。尚未运行任何 GPU/PIM operator 或 C++ simulator。未将这些 launch/import 失败的 wall time 报成 simulator runtime。

完整 argv、cwd、stdout、stderr、exit/launch error、wall_seconds 在 `results/build_attempts.json` 和三个逐命令 JSON 中；这些是程序捕获日志，不是模拟数据。未继续修改以跳过依赖，因为另有 pandas/ruamel、Python 和 C++ binary 缺失，无法产生真实原始 baseline。

## 恢复所需

需要可用的 Python3.12、uv（或等价的可复现依赖环境）、cmake>=4.0.3 与 ninja，以及上游锁定依赖。CMake 会通过 CPM 下载固定 Ramulator2 提交 `42b625986b0b9f8f0635608944f2e66ca58b0c66`，CPM 自身也需下载；本地没有对应依赖缓存。仅有 NeuPIMs 的 NewtonSim 不能替代它。

恢复环境的正常网络，或提供包含这些完整依赖的离线环境/缓存。依赖就绪后严格回到未改动原始 baseline：

```bash
uv sync
uv run cmake --preset release
uv run cmake --build build
```

然后使用 docs/api.md 的 Performance API 跑小样本（建议 batch8、lin16、lout2；原先 batch1 尝试未过 import，不足以验证该尺寸 trace 可用），再固定正式实验参数。不要使用 scripts/run.py 无差别运行全部 accuracy/性能实验。

每个 experiment 用独立进程，遵循 docs/reproducibility.md 的缓存隔离；所有 baseline 结果写入唯一目录，记录 commit、配置、binary/hash、逐 category 数据、stderr 和 wall time，完成后计算哈希，不覆盖已有 baseline。

## 待执行矩阵（没有结果）

目标模型 mamba2-2.7b；system=Pimba；A100×1，5 HBM stacks、40 channels、HBM1512 MHz，GPU dbyte2、PIM state_dbyte1、chunk group/command scheduling 开启。正式 lin/lout=2048/2048 来自原 API 默认值。batch32/64/128；sub-batch8/16/32/64，排除大于 batch 的组合，共11组 modified，另3组 baseline。每组配置完全保存，分区/共享资源策略和开销参数还需实现后确定。

## 指标定义（拟用，尚无数值）

原始 Mamba2 只覆盖 lout-1 个 decode steps，无 prefill。设 D=lout-1、T为这些 steps 全部 sub-batch 完成的 simulated makespan：decode throughput=B*D/T tokens/s；decode batch completion latency=1000*T ms；平均 decode step/TPOT=1000*T/D ms/token。不能标成完整 end-to-end latency。

每 request/sub-batch 的 completion latency 应由其最后一个 token 的完成事件给出；不是 T/B。单 batch 内请求平均完成时间也不等于 TPOT。modified 应报告 makespan和completion分布，不能对有重叠的 operation time 求和。

Speedup=modified throughput/baseline throughput；latency reduction=(baseline T-modified T)/baseline T*100%。GPU/PIM busy time 用对应资源区间的并集、idle=T-busy；overlap用两资源忙区间交集/T。memory utilization需 traffic/capacity/accounting 的来源，不能凭 busy 时间推断。图表只允许从真实执行记录生成。
