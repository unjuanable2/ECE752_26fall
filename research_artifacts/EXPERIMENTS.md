# 实验配置、执行记录与复现

## 实际完成

原始 Pimba `0a902eeafddbfddaa34723ed74a761e4453b870d` 已成功编译运行；NeuPIMs `f299af3fc8f20077e816f2e48313294cddb4bd7c` 只读参考。Pimba 子仓库分支 `feature/sub-batch-interleaving`；未 push。

旧失败日志和报告保留。本次网络正常，依赖安装在 workspace 内，无 sudo、额外凭据或付费资源。使用 CPython3.12.16、CMake4.0.3、Ninja1.13.2、GCC11.4，依赖精确版本见 `pimba/scripts/research/requirements-performance.lock`，环境与 binary hash 见 `results/run-20261028/environment.json`。

未运行上游 `uv sync` 的完整 accuracy 环境：performance 不需要 PyTorch、Triton、lm_eval 或模型权重。原始 baseline harness 只绕过顶层包对 accuracy/draw 的急切导入，实际 performance 和 C++ 源码保持原样；日志记录 `source_diff=""`，原始基准目录仍保留在 `results/run-20261028/pristine-pimba`。修改后顶层 lazy import 支持正常 `from pimba import Performance`。

## 固定配置

| 参数 | 数值 |
|---|---|
| 模型 | mamba2-2.7b：64 decoders，hidden2560，heads80，head_dim128，state_dim64 |
| GPU | A100×1；108 SM，312 TFLOP/s，1.935 TB/s，compute util0.8/memory util0.85，上游 tile/cache 模型 |
| 总 HBM | 5 stacks，40 channels，1512 MHz，原 LAX/dramsim3 timing |
| PIM | Pimba SPU/MX8，state_dbyte1，command scheduling与chunk group开启 |
| GPU 精度 | dbyte2；所有对照保持不变 |
| 主矩阵 lin/lout | 2048/2048，实际 2047 decode steps；没有 prefill |
| batch | 32/64/128 |
| sub-batch | 8/16/32/64，排除大于 batch，共11组合 |
| modified mapping | GPU20/PIM20 channels，总数40；sync1000ns/每 handoff |
| 附加实测 | 同分区串行对照；sync0/5000ns；PIM16/24 channels敏感性 |

主实验使用 `full2048/`；`run-20261028/` 顶层为先完成的 lout17 验证矩阵和 timeline。smoke baseline batch8、lout2。不会将两种长度的 latency 混用。每个实验使用独立进程；无跨实验缓存复用。

## 环境重建

从 `/home/jianing/ECE752` 执行；已有环境不必重装：

```bash
python3 -m pip install --target .research-tools uv==0.13.6
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.python" .research-tools/bin/uv venv --python 3.12.16 .venv-research
UV_CACHE_DIR="$PWD/.cache/uv" .research-tools/bin/uv pip install --python .venv-research/bin/python -r pimba/scripts/research/requirements-performance.lock
```

在 `pimba/` 下构建原 C++ 模拟器（从始至终没有修改 csrc）：

```bash
PATH="$PWD/../.venv-research/bin:$PATH" cmake --preset release
PATH="$PWD/../.venv-research/bin:$PATH" cmake --build build -j 4
```

CPM 固定下载 Ramulator2 `42b625986b0b9f8f0635608944f2e66ca58b0c66`，其固定依赖由源码 CMake 获取。实际 build 输出保存在 `results/run-20261028/build.log`。

## 重新运行原始 baseline

下面使用新的输出目录，避免覆盖已保存 baseline。原始 harness 强制检查 commit 与 src/csrc 洁净状态。准备独立原始 checkout（也可使用已经存在的 pristine-pimba）：

```bash
git clone --no-hardlinks pimba /tmp/pimba-original-repro
git -C /tmp/pimba-original-repro checkout --detach 0a902eeafddbfddaa34723ed74a761e4453b870d
ln -s "$PWD/pimba/build" /tmp/pimba-original-repro/build
mkdir -p results/repro/full2048
for b in 32 64 128; do
  PIMBA_ORIGINAL_ROOT=/tmp/pimba-original-repro .venv-research/bin/python pimba/scripts/research/original_baseline.py "$b" "results/repro/full2048/baseline-b${b}.json" 2048
 done
```

本次实际先在未修改的 pimba 源码上运行 batch8 smoke 和 batch32/64/128、lout17 baseline；修改后从 pinned pristine checkout 跑 full2048 baseline。同一 binary hash 为 `e64b1f7f13f001bc7d4a633ea51f54833ae9b2bc6095753c91d95706671a98a5`。`BASELINE_SHA256SUMS` 检查结果文件；所有输出脚本遇到已有目标会拒绝覆盖。

## 新调度器、测试与图表

```bash
PYTHONPATH=pimba/src .venv-research/bin/python -m unittest discover -s pimba/tests -v
.venv-research/bin/python pimba/scripts/research/run_matrix.py --lout 2048 --output-dir results/repro/full2048
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv-research/bin/python pimba/scripts/research/summarize.py --input-dir results/repro/full2048 --figures figures/repro
```

`run_matrix.py` 逐一启动独立 interpreter，完成11组合及5个控制/敏感性实验；已有结果会跳过。每个 case JSON 记录 command、UTC、runtime、配置、profile 和计数；原始 PIM 文件在同名 `*-pim/`。可单独执行：

```bash
.venv-research/bin/python pimba/scripts/research/run_case.py --batch 128 --sub-batch 32 --lout 2048 --no-timeline --output results/repro/single.json
.venv-research/bin/python pimba/scripts/research/replay_pim.py results/run-20261028/baseline-b128-pim
.venv-research/bin/python pimba/scripts/research/replay_pim.py results/run-20261028/modified-b128-s32-pim/a132a351d2a8e7f5
```

full2048 case 默认在矩阵中禁用逐事件日志，仍完整执行每个事件并累计统计/检查 state；避免数百万节点的 JSON。timeline 图来自 lout17、B128/S32 的完整已保存事件。所有图表从真实 JSON 生成，没有填充缺失实验数据。

## 指标定义

令 D=lout-1，B=batch，T为全部 sub-batch 完成 D 个 decode steps 的 makespan：吞吐量 `B*D/T` tokens/s；decode batch completion latency=`1000*T` ms；平均 TPOT=`1000*T/D` ms/token。原始 Mamba2 不使用 lin，不包含 prefill、首 token、arrival/queueing、sampling，故不能称完整 inference end-to-end latency。

modified 每 request latency 用其所属 sub-batch 的最后 token 完成事件（所有请求同时到达）；输出按 size 加权平均的 completion_ms，不是T/B。TPOT只是 batch平均步长，不是每个 request 的 inter-token gap 分布。

GPU/PIM utilization 是 operation busy interval 的占比，不是实际 SM occupancy；idle=T-busy，overlap为两资源忙区间交集/T。GPU staging 也保守占用GPU资源。GPU HBM利用率=模型GPU traffic/(分区peak bandwidth*T)，仅GPU分区；没有报告全系统实测带宽或PIM内部带宽利用率。GPU clock未定义，所以不伪造GPU idle cycles；JSON报告秒，可按HBM1512MHz换算bus周期，但不是GPU时钟周期。

所有数值均为模拟器执行结果，而非真实 GPU 硬件 benchmark。系统模型限制见 IMPLEMENTATION.md。
