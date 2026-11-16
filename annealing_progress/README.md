# 中间尺度跨区：研究补充包

先读 `progress.md`。本文的命题、推导、数值结果和限制均在该文件内；`figures/mechanism_and_risk.png` 是主要概览图，PDF 为同一科学绘图的矢量版本。

## 运行

证书只依赖 Python 标准库。交叉检查、实验、作图需要 NumPy 和 Matplotlib；实际运行版本见 `results/environment.json`。

在本目录执行：

```bash
python3 code/check_implementation.py
python3 code/certificate.py --noise .005 --out certificate_reproduction.json
OPENBLAS_NUM_THREADS=1 python3 code/experiments.py --stage all --n 10000
OPENBLAS_NUM_THREADS=1 python3 code/fair_controls.py --n 10000
python3 code/plot_progress.py
```

可选：通过旧全二维网格包围法复核相同进展不等式：

```bash
python3 code/certificate.py --noise .005 --progress-method grid --out certificate_grid_reproduction.json
```

不要使用 `python -O`：区间验证器依赖断言，主动拒绝关闭断言的运行方式。无需重新生成 `results/candidates.json`；它只是预热和两类轨迹的待验证中心，验证器会重新包围其残差。

## 文件

| 文件 | 作用 |
|---|---|
| `code/interval.py` | 50 位 Decimal、向外舍入的区间核；未形式化验证 |
| `code/certificate.py` | 共同端点错误球、矩形不变性、有限步进展、终端球及证明拼接 |
| `code/check_implementation.py` | 有理参考算术和模型恒等式检查；不代替证明 |
| `code/model.py` | 独立浮点算法实现 |
| `code/make_candidates.py` | 可选数值候选生成；不属于可信证明步骤 |
| `code/experiments.py` | 固定预热相图；验证集选参数；多噪声水平风险实验 |
| `code/fair_controls.py` | 固定终端 20 步相图、固定有效系数消融、新独立测试、选择后恢复诊断 |
| `code/plot_progress.py` | 从结果重建概览科学图 |
| `results/certificate_scalar.json` | 当前标量进展方法的完整区间证书 |
| `results/certificate_rechecked.json` | 修改前全二维进展方法的完整复跑证书 |
| `results/implementation_checks.json` | 当前实现检查结果 |
| `results/risk.json` | 独立验证/测试、共同参数和分别调参结果 |
| `results/fair_controls.json` | 新增独立测试和对照；补充稿主风险表来自此文件 |
| `results/phase.npz` | 原固定预热 12 步相图 |
| `results/fixed_tail_phase.npz` | 固定终端 20 步相图 |
| `results/nominal.json` | 端点预热扫描、名义轨迹和误差 |
| `results/candidates.json` | 固定有理候选；仅用于有限轨迹误差管 |
| `source/draft.md` | 本轮输入初稿的副本，仅供对照；其中旧外链及历史验证器不在本包中 |
| `SHA256SUMS.txt` | 打包文件校验摘要，不含该摘要文件自身 |

所有图和 Monte Carlo 结果均是浮点探索。完整参数盒的认证范围只以 `progress.md` 第 1、2、8 节和对应证书为准。不能把图中的名义成功区域当成整片区间认证，也不能把模板真值的证书套到连续 GMM 的高斯尾部。

新结果分开保存，没有覆盖输入初稿。论文的新颖性排查、结构化参数族外推、真实视觉任务以及学习去噪器误差容忍量仍未完成。
