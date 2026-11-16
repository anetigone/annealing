# 中间尺度校准：新的理论补充与复现材料

阅读顺序：先看 `scale_calibration.md`；需要原模型的完整风险拼接时，再看 `baseline/theory_chain.md`。

本包包含两项结果草稿：

- 固定宽度的中间方差尺度带 `[0.69,0.71]`，配合可观测目标区域与入口检测，最多 25 次中间调用后进入正确终端吸引域，保留渐近 oracle 风险。
- 对所有观测共用且充分停留的局部平台，偏离原平衡尺度达到 `20 alpha` 会使至少一类永久选错。因此原来的线性尺度校准精度在这个协议内有必要性。

新上界明确使用观测判别来控制尺度；恢复输出仍由原始 RED 更新产生。前稿的对数跨区下界不能扩张到全部可观测尺度策略。噪声耦合、已知几何、精确去噪模型与终端系数窗口仍是条件。

## 验证及复现

```bash
python3 code/calibration_certificate.py
OPENBLAS_NUM_THREADS=1 python3 code/diagnostics.py --n 3000
```

区间验证只需 Python 标准库；数值诊断需要 NumPy。禁止使用 `python -O`。环境版本、种子与运行规模见 `results/environment.json`。

| 文件 | 作用 |
|---|---|
| `scale_calibration.md` | 定理、证明、协议边界与论文表述建议 |
| `code/calibration_certificate.py` | 整个参数盒的辅助区间不等式 |
| `code/coarse_control.py` | 可观测尺度控制与同一 RED 更新 |
| `code/diagnostics.py` | 7 组调用量、240 个扰动实例、9 组尺度敏感性及连续风险诊断 |
| `results/calibration_certificate.json` | 区间不等式记录 |
| `results/diagnostics.json` | 浮点诊断记录 |
| `baseline/` | 本轮依赖的上一版推导、几何代码和辅助常数 |

这些辅助证书和诊断不能代替完整解析审阅或形式化验证。固定点只在解析存在性与比较论证中出现，验证器不导入数值驻点候选。

`SHA256SUMS` 固定了本次材料的内容摘要。
