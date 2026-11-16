# 原始 RED：从几何到计算与风险的理论包

先读 `theory_chain.md`。这是一份新结构化三模板族上的计算机辅助定理草稿，包含三个相互衔接的结果：

1. 两段受限尺度集合内，任意长、任意顺序、可依赖观测的切换仍然失败；由几何求出的中间尺度解除障碍。
2. 证明实际失败的跨区步数下界，以及跨区、终端细化上界；规定两阶段协议内的主要阶数匹配。
3. 同一参数族中，原始 RED 的无条件风险渐近达到已知分量基准，受限尺度策略保留非消失风险。

关键范围：弱观测特征值趋零，同时分量与观测噪声缩小；中间窗口及终端系数偏差窗口宽度随该特征值缩小。这不是任意 GMM、任意算法或固定噪声下的结论。辅助区间不等式已经检验，完整解析拼接及自建区间核仍需独立审阅。

## 文件与证据层次

| 文件 | 作用 | 不能据此声称的内容 |
|---|---|---|
| `theory_chain.md` | 参数构造、三个定理、证明、界限及风险拼接 | 尚未经过独立审稿或形式化验证 |
| `code/structural_certificate.py` | 整个参数盒上的责任、角点、收缩及终端辅助不等式 | 程序本身不验证全部书面推理 |
| `code/interval.py` | 50 位 Decimal 向外舍入算术 | 不是第三方形式化验证内核 |
| `code/check_family.py` | 5,400 个浮点恒等式与导数交叉检查 | 不替代区间证明 |
| `code/experiments.py` | 步数、200 个随机实例、每类 3,000 次风险诊断 | 不证明全参数结论或真实临界相图 |
| `results/*.json` | 本轮运行的原始结果 | 数值误差达到机器精度时须谨慎解释 |
| `figures/theory_chain.png`、`.pdf` | 同族上的调用量与风险图 | 没有声称充分界发散是真实分岔律 |

## 复现

在本目录执行：

```bash
python3 code/structural_certificate.py
python3 code/check_family.py
OPENBLAS_NUM_THREADS=1 python3 code/experiments.py --n 3000
python3 code/plot_results.py
```

第一个程序只需 Python 标准库；其余需要 NumPy、Matplotlib。不要使用 `python -O`。环境版本记录在 `results/environment.json`。证书输出包含 alpha 的闭包端点 0，仅用于统一辅助常数；正文定理要求 alpha 严格为正。

`SHA256SUMS` 记录打包前的文件摘要，便于确认后续审阅与当前版本使用相同文件。
