# 实验

[独立带噪调参与同门控预算对照](results/budget_control_validation/analysis.md)包含冻结计划、完整输出和复跑命令；入口为 `code/budget_control_validation.py`，汇总绘图为 `code/summarize_budget_control.py`。复跑必须指向新目录。

[就绪差距与带噪选参失效的原因研究](results/budget_control_mechanism/analysis.md)链接原轨迹重放、校准影响、分界积分及其复算命令。

正式稿三模板机制图与小规模配对诊断：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/mechanism_diagnostics.py --out experiments/results/mechanism_reproduction --n 1024
```

参数和种子固定，没有搜索；重复运行请更换输出目录。结果及适用边界见 [机制诊断分析](results/mechanism_review/analysis.md)，辅助区间复跑入口见 [evidence/README](results/mechanism_review/evidence/README.md)。

统一主稿的新结构化族实现保留在 [几何与风险代码](../red_geometry_computation_risk/code/)和[观测触发尺度控制代码](../red_scale_calibration/code/)，对应交付说明见[结构化族说明](../red_geometry_computation_risk/README.md)和[尺度控制说明](../red_scale_calibration/README.md)。这些入口默认输出到各自 results；复跑应使用隔离副本，保留交付证据。已有本机复核见 [尺度控制复核结果](results/calibration_review/)。下列 code/ 入口用于原固定几何族。

实现位于 `code/`，原始输出位于 `results/checkpoint/`；CPU/float64，不依赖外部数据或训练权重。当前结论与限制见 [status.md](../status.md)。

## 环境

本机 PowerShell 使用：

```powershell
& H:/anaconda3/envs/py312/python.exe -m pip install -r experiments/requirements.txt
```

requirements 保留原包的精确版本；整理工作未修改本机环境，也不要求覆盖现有环境。其它机器使用 Python 3.12。

## 从工作区根目录复现

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/r1_experiment.py --out experiments/results/reproduction --samples 200000
& H:/anaconda3/envs/py312/python.exe experiments/code/r1_three_templates.py --out experiments/results/reproduction
& H:/anaconda3/envs/py312/python.exe experiments/code/r1_three_geometry.py --out experiments/results/reproduction
```

三个脚本默认输出均基于脚本位置解析到 `experiments/results/reproduction/`，不依赖当前目录。显式 `--out` 相对调用目录解析。几何脚本读取同一输出目录的 selected JSON 和 validation CSV，需先运行三模板脚本。重复使用同一输出目录会覆盖其中同名文件，新的参数批次请另取目录。

- `r1_experiment.py`：双模板公式校验、证明常数、成对噪声实验、调度和先验权重扫描、初始化对照。
- `r1_three_templates.py`：固定种子的 300 设置探索筛选和独立噪声验证（每设置 100,000 样本）。
- `r1_three_geometry.py`：终端继续迭代、驻点分支采样、吸引域诊断与绘图。

## 已有证据

`results/checkpoint/` 整体保留原包数据。双模板见 `summary.json`、`paired_2d_results.csv`、`initialization_check.csv`；三模板见 `three_template_selected.json`、`three_template_search.json`、`three_template_validation.csv`；几何诊断见 `geometry_summary.json` 及对应 CSV/NPZ/PNG。512 步终端继续迭代属于诊断，不计入 64 步主比较。

## 三模板证明验证

有限预算和几何证书分别位于 `code/certificate/`、`code/geometry/`。每组保持原有模块名与直接运行入口，避免混用两套区间核。验证仅依赖 Python 标准库，必须开启断言（不使用 `-O`）。

从工作区根目录执行：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/certificate/verify.py
& H:/anaconda3/envs/py312/python.exe experiments/code/geometry/verify_geometry.py
```

默认读取 `results/certificate/candidates.json` 或 `results/geometry/sparse_candidates.json`，分别输出到 `results/certificate_reproduction/` 和 `results/geometry_reproduction/`。路径基于脚本位置解析，支持任意当前目录。可用 `--data <候选所在目录>` 和 `--out <输出目录>` 改写；显式相对路径基于当前目录。`verify.py --skip-branch` 仅检查有限轨迹，不作为完整分支证书。

原始证据目录包含候选、证书 JSON、逐步或逐段界、图示及原包装校验记录。其源码哈希只代表原交付版本，整理后的代码不以哈希一致性为门槛，也不默认重算哈希。先前本机复核输出保存在 `results/certificate_review/` 与 `results/geometry_review/`。

可选补充检查与绘图：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/certificate/check_implementation.py
& H:/anaconda3/envs/py312/python.exe experiments/code/certificate/make_figure.py
& H:/anaconda3/envs/py312/python.exe experiments/code/geometry/check_geometry.py
& H:/anaconda3/envs/py312/python.exe experiments/code/geometry/plot_geometry.py
```

这些入口也支持 `--out`，默认写入对应 reproduction 目录。补充检查与绘图依赖 NumPy/SciPy/Matplotlib；可选版本范围见 `requirements_certificate.txt`。绘图不承担证明。

候选生成入口为两组中的 `make_candidates.py` / `make_sparse_candidates.py`，默认也写入 reproduction 目录。重新生成后用 `verify.py --data experiments/results/certificate_reproduction` 或 `verify_geometry.py --data experiments/results/geometry_reproduction` 验证新候选；不覆盖原始候选。候选生成不属于验证步骤，本次整合未运行。


## 可计算尺度几何证书

从根目录运行（标准库，无参数网格；失败在 `-O` 下仍抛异常）：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/scale_geometry/certificate.py --precision 60 --output experiments/results/scale_geometry_reproduction/certificate_60.json
& H:/anaconda3/envs/py312/python.exe experiments/code/scale_geometry/certificate.py --precision 90 --output experiments/results/scale_geometry_reproduction/certificate_90.json
```

原交付与两份输出保留于 `results/scale_geometry/`，新运行使用独立 reproduction 目录。两种精度不是两套独立实现。解析归约见 `paper/latex/scale_geometry_proof.tex`。


## 当前尺度协议的配对诊断

从根目录运行，输出指向一个尚不存在的目录。

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/scale_availability_diagnostic.py --out experiments/results/scale_availability_reproduction
```

[现有诊断及协议](results/scale_availability/analysis.md)


## 投稿前尺度几何交叉复核

交叉实现依赖 mpmath。命令从工作区根目录运行，输出到新文件。

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/scale_geometry/certificate_crosscheck.py --out experiments/results/scale_geometry_submission_reproduction/crosscheck.json
```

[已归档的复核范围与结果](results/scale_geometry_submission_review/review.md)
