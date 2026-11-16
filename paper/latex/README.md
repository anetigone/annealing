# CVPR 2027 manuscript

使用根目录 `author-kit-CVPR2027-v1-latex/` 的原版 cvpr.sty 与 ieeenat_fullname.bst，匿名 review 模式。原模板未修改。当前研究进展与提交前待办只见 [status.md](../../status.md)；中文推导入口仍为 [draft.md](../draft.md)。

在本目录运行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

`main.pdf` 是英文论文稿，`supplement.pdf` 是单独的推导补充。作者信息未提供，使用匿名模式。模板差异见 [比较记录](../notes/template_comparison.md)。

补充稿的机制图来自[小规模机制诊断](../../experiments/results/mechanism_review/analysis.md)。[随稿证书包](evidence_revision.zip)包含主参数盒、宽噪声裕量、名义方差根、平衡族和共同切换的脚本、依赖、复现命令及预期输出。请在新解压目录复跑。


[风险分离主结果](scope_extensions.tex)、[非对齐范围](rotated_scope.tex)、[一般证明接口](reachability.tex)、[双模板端点结论](two_endpoints.tex)和[三模板尺度判据](scale_geometry.tex)按此顺序接入正文。补充 S4–S7 集中给出噪声、参数盒、端点和方差带证明，S8 起为平衡子族。[精细校准与敏感性诊断](../../experiments/results/scale_availability_revision/analysis.md)记录表格数据与复跑命令。

正文第 7 节的证明链见 [recovery_argument.tex](recovery_argument.tex)。主文图 1 由 [plot_reachability.py](../../experiments/code/plot_reachability.py) 生成，运行参数为 `--out paper/latex/figures/reachability.pdf`。补充中的名义角点尺度图由 [plot_variance_band.py](../../experiments/code/plot_variance_band.py) 生成，运行参数为 `--out paper/latex/figures/variance_band.pdf`。两条命令均从工作区根目录运行。

补充 Figure 2 来自[跨几何诊断](../../experiments/results/geometry_prediction_wide/analysis.md)。新增的[精度参考](../../experiments/results/precision_revision/analysis.md)、[弱正则预算](../../experiments/results/weak_budget_revision/analysis.md)和[真实照片 patch 设置](../../experiments/results/patch_revision/analysis.md)分别记录命令与数值范围。

补充 S22–S24 的一般 GMM 收缩、系数分解与多竞争阈值分别见 [terminal_history_proof.tex](terminal_history_proof.tex) 和 [multi_competitor_proof.tex](multi_competitor_proof.tex)。[终端与阈值证据包](terminal_evidence.zip)包含预期输出、依赖版本和可独立解压运行的验证入口。其[认证范围](../../experiments/results/patch_terminal_pairwise/analysis.md)以导出的模型系数和状态作为精确输入。

补充 S25 的[冻结独立验证](independent_patch_validation.tex)对应[预定设置、逐图结果与命令](../../experiments/results/patch_independent_validation/analysis.md)。图像来源见[照片说明](../../experiments/results/patch_independent_data/provenance.md)。

[独立照片复跑包](independent_patch_evidence.zip)包含冻结模型、完整轨迹快照、候选球、区间结果和命令，可在新解压目录直接验证。

补充 S26 的[切换反例](raw_patch_switching.tex)对应 [patch 37 邻域证书](../../experiments/results/raw_patch_neighborhood/analysis.md)和[冻结调度噪声复验](../../experiments/results/raw_patch_substitute_validation/analysis.md)，两份记录各含数据或证书以及复跑入口。

主文图 2 从冻结的预算选带计划与逐模型结果生成。在工作区根目录运行 `& H:/anaconda3/envs/py312/python.exe experiments/code/plot_scale_selection.py --data experiments/results/scale_selection_budget --out paper/latex/figures/scale_selection.pdf`。

补充 S27 给出[固定弱增益的非对齐证明](rotation_proof.tex)及标量复跑入口。补充 S28 记录[选带、独立调参和认证步数复验](scale_selection.tex)。[角度与平台证据包](scale_control_evidence.zip)包含脚本、冻结输入、结果和命令；见[复跑说明](../../experiments/results/rotation_certificate/reproduce.md)和[逐模型分析](../../experiments/results/scale_selection_followup/analysis.md)。
