# Theory, optimization, and statistical learning for vision

## 正式稿引用补充（2026-11-02）

本轮把英文稿实际引用从 5 篇扩至 15 篇，正文比较见 [related_work.tex](../latex/related_work.tex)，完整书目见 [references.bib](../latex/references.bib)。未以引用数判定新颖性，也未开展完整前向引文扫描。

- PnP 起源采用 [Venkatakrishnan–Bouman–Wohlberg 作者主页](https://brendt.wohlberg.net/publications/venkatakrishnan-2013-plugandplay.html)给出的 Purdue technical report 身份；不把该页面的报告书目冒充会议书目。补入 [Chan–Wang–Elgendy 2017](https://arxiv.org/abs/1605.01710) 与 [Ryu et al. ICML 2019](https://proceedings.mlr.press/v97/ryu19a.html) 的固定点/条件收敛背景。核摘要、元数据，未完整审其证明。
- 退火近邻采用 [Hurault v2](https://arxiv.org/html/2609.32393v2)，核正终端尺度、终端驻点与 §5 的 stationary-point selection 局限；[SNORE ICML 2024](https://proceedings.mlr.press/v235/renaud24a.html)采用正式会议信息并据摘要陈述其随机正则与退火分析。DPIR 引用作者提供的 [TPAMI 正式 PDF](https://ofsoundof.github.io/assets/pdf/2022_tpami_DPIR.pdf)，仅用作深度去噪器与调度实践背景，不替代本稿算法基线。
- 经典 continuation 除原 Blake–Zisserman 和 mean-shift 两篇外，补入 [Hazan–Levy–Shalev-Shwartz ICML 2016](https://proceedings.mlr.press/v48/hazanb16.html)：核其特定非凸函数族的全局保证；尚未逐条件证明本稿不被其覆盖，不作新颖性排除结论。
- [GAMMA v1 §4.1](https://arxiv.org/html/2609.37529v1)明确 H1 凸数据项、H2 对数凹先验；本稿据此区分协议与先验条件。[Leong–Traonmilin](https://proceedings.mlr.press/v300/leong26a.html)已发表于 AISTATS 2026，改用正式 PMLR 300:1117–1125；其 varying projection、RIP 和低秩 GMM 恢复结论已存在。新增 [Efron 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325056/)作为 Tweedie 公式出处，明确不是本稿新工具。

OpenAlex 读取环境变量凭据直接核对四个 arXiv DOI，不记录凭据：2609.32393 → W7214795678；2609.37529 → W7214928169；2509.20511 → W4414566415；2402.01779 → W4391556969。四条标题和作者均与原文一致；OpenAlex 中 Leong 条目年份 2025 是预印本年份，不覆盖正式会议 2026 年份。这里只核元数据，不以索引推导技术结论。

下文保留 2026-10-28 的历史检索正文，其版本与会议身份以本节和正式书目更新为准。完整定理级查新、经典 GNC 逐条件对照和独立证明审查仍未完成。

检索日：2026-10-28。主查2024年至今，特别检查2026年及10月新稿/修订。沿用 SEARCH_PROTOCOL v3 的问题清楚、边界明确、证明入口、视觉意义四项标准。本轮扩大主题、比较新入口，不是对旧A1/T1/S1的完整更新。没有实验或新定理。

## 结论

优先继续查证 **R1：退火去噪的驻点选择与图像恢复质量**；次选 **R2：近似 c-rectified flow 的有限预算误差**。两张均为待核卡，0个确认通过全部新颖性门槛的题目。建议先花小预算查重和反例计算，不直接投入完整证明。

相比泛泛“视觉学习的泛化界”，两个入口能直接连接可观察行为：修复落入错误图像模式，以及少步生成/重流的误差累积。3DGS可辨识性作为横向补漏，暂不另立第三张卡。

本轮整理15篇arXiv种子，逐篇API核元数据和最新版本；部分读关键原文，其余仅摘要/设置。引用数没有查询，全部null；不以新稿引用少降级。来源链接与作者完整列表见 papers.json。另有官方会议和期刊线索用于查重，未全部列入种子计数。

## 文献地图与阅读边界

| 文献 | 日期/核验版本 | 已知内容与本轮判断 |
|---|---|---|
| [Convergent Plug-and-Play Image Restoration with Annealed Noise Levels](https://arxiv.org/html/2609.32393v2) | 2026，v2；API日期见JSON | 核§3的驻点结果与§5局限，v1先读后补开最新v2；未逐行比较版本。正噪声下限、结构化去噪器等条件不可省略。退火收敛本身已不足为新题。 |
| [Principled MAP estimation…](https://arxiv.org/html/2609.37529v1) | 2026-09-29，v1 | 核§4 H1/H2、迭代与调度条件、§5/6。GAMMA在特定凸/对数凹条件下追踪MAP；不可从图像实验推出任意多模态先验的全局保证。 |
| [A Recovery Theory for Diffusion Priors](https://arxiv.org/html/2509.20511v1) | 2025-09-24，v1 | 核§3–4、凸集/LR-GMM与投影误差递推入口。RIP型观测条件和模型集关键；既有恢复结果不能被“扩散先验没有恢复理论”覆盖。 |
| [c-Rectified flow](https://arxiv.org/html/2608.02487v2) | 2026-08初稿；**10-26 v2** | 核§3.1、3.4 Assumptions3.13–3.15、Thm3.16–3.18、App.F.4入口。速率依赖投影稳定性；局部成立不等于所有迭代自动保持。 |
| [Minimax-Optimal Predictive Flow Matching under Distribution Drift](https://arxiv.org/html/2610.28898v1) | **2026-10-20**，v1 | 核§3 Assumption4和拟合误差说明。非凸函数类的统计遗憾结果保留近似全局拟合接口，不是任意神经优化器收敛。 |
| [Diffusion Models Are Statistically Optimal…](https://arxiv.org/html/2605.30153v1) | 2026-05-28，v1 | 核§3 Thm1的精确子空间恢复事件、样本分割与采样结论。统计构造不等于U-Net梯度训练保证。 |
| [Flow matching achieves almost minimax optimal convergence](https://arxiv.org/abs/2405.20879) | 首发2024；最新v2 | 摘要/元数据；[ICLR2025官方](https://proceedings.iclr.cc/paper_files/paper/2025/hash/44a427b77e9727fcec560e2f8d6925e0-Abstract-Conference.html)确认。已有Wasserstein统计率，不能再笼统立minimax题。 |
| [Flow Matching is Adaptive to Manifold Structures](https://arxiv.org/abs/2602.22486) | 2026，v2 04-09 | 元数据/摘要：流形支撑与内禀维数已有分析，未审证明。 |
| [Robustness of Diffusion Models under Distribution Shift](https://arxiv.org/abs/2609.27546) | 2026-09-23，v1 | 摘要：Wasserstein偏移的稳健score估计与采样代价已有分析，未审定理。 |
| [Multi-Objective Learning for Diffusion Models…](https://arxiv.org/abs/2605.25210) | 2026-05-24，v1 | 摘要：专家到通用模型蒸馏、有限配对数据界，视觉恢复/机器人验证。未审证明，不纳入主卡。 |
| [Diffusion Models with Heavy-Tailed Targets](https://arxiv.org/abs/2601.06715) | 2026-01-10，v1 | 摘要：重尾score估计/采样界已有，不以“去掉轻尾”泛泛立题。 |
| [When 3D Gaussian Splatting Recovers Real Surfaces](https://arxiv.org/html/2608.30054v1) | 2026-08-30，v1 | 核Thm7.1的假设依赖与正文入口；有限角向容量/错误几何解释已有。尚未审完整证明或后续，不把角容量与几何的关系当新发现。 |
| [Opt3DGS](https://arxiv.org/abs/2511.13571) | 首发2025，v1 | API/摘要，3DGS优化方法近邻；[AAAI2026官方PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37438/41400)发现，尚未审理论。 |
| [Picard Proximal Monte Carlo…](https://arxiv.org/abs/2608.17666) | 2026-08-18，v1 | 摘要：非对数凹、近似score及退火的后验采样近邻；未审全文。采样与MAP优化要分开。 |
| [From stability of Langevin diffusion…](https://arxiv.org/abs/2505.14177) | 首发2025，最新v2 | 官方论文摘要/元数据：无穷远强凸等条件下非凸采样稳定性；不能与任意多模态全局MAP等同。 |

## R1 待核卡：退火怎样改变错误模式的吸引域？

**分类：作者明确提出的驻点选择问题，下面的具体命题为本轮拟定。** 来源为Hurault §5；论文已有站稳于驻点的结论，尚不能据此解释恢复质量。这里不将作者的future work当作截至今日仍开放的证明。

**对象与命题。** 取两至三个固定低维图像模板组成的Gaussian mixture先验p，线性模糊/缺失观测 y=Ax*+η，η为已知高斯噪声。固定确切RED-GD：x_(k+1)=x_k−γ[∇f_y(x_k)+λ(x_k−D_σk(x_k))]，D是该p的精确MMSE去噪器；这里残差未除以σ²，不与GAMMA混写。比较相同初始化分布、相同预算K及相同终端ε的常值与退火调度。寻求一个非空、非退化的模板/观测参数族，给出可显式估计的速度阈值，使P(输出进入正确模板邻域)相对常值调度有严格增量；给出过快/过慢及错误先验权重导致反转的反例。邻域半径、成功概率和观测噪声均需写定。

**缺口和证明入口。** 研究临界点分岔、吸引域边界及其随σ的移动；用延迟分岔/离散动力系统跟踪分析，解释如何越过错误模式、又何时会被高权重错误模式捕获。最后将模式归属转换成重建平方误差界。不是把条件“迭代一直在好盆地”当结论，也不能只做一次下降引理。

**视觉联系与最小验证。** 对有歧义的纹理修复/遮挡补全，比较同一最终目标下不同调度的模式选错率与恢复误差，先精确二维/低维计算，再做32×32模板实验。固定测量噪声、调用次数和初始化；真实标签仅离线评价。单卡5090无需大规模训练，尚未实验。

**近邻与阻断。** GAMMA的对数凹极限与本题多模态不同；恢复理论已有LR-GMM结果，不得重复其观测/初始化条件。还需核经典deterministic annealing、graduated nonconvexity、homotopy mode tracking，以及Hurault最新版全部附录和同作者后续。SNORE原论文已给近似去噪误差处理，不能只加有界误差项算贡献。已有[线性PnP收敛正则化](https://link.springer.com/article/10.1007/s10208-024-09654-x)亦须区分优化误差与恢复误差。

**停止条件。** 结果只是标准双井跟踪的换名；结论依赖目标标签调参；匹配预算后无可重复模式增量；只有均方误差变化而没有预测到吸引域变化；或者既有LR-GMM/continuation定理直接覆盖。此卡仍未过查新门槛。

## R2 待核卡：近似重流何时抵消理论收缩？

**分类：从定理边界推断。** 来源c-Rectified flow v2 §3.4；它已有理想算子的条件性收缩，不能重做“reflow不一定最优”或Gaussian精确算例。

**对象与命题。** 固定二次代价及一族具有明确密度上下界/光滑性条件的源目标分布，使用标量势神经网络参数化梯度投影场；第k轮有限样本拟合后用N_k步ODE得到近似端点律Q_k。将真实生成误差W2(Q_k,Q)与相对最优运输代价同时作为目标；当端点边际已偏离时不能继续把代价差称为非负“excess OT cost”。目标是由数据量n_k、可证的拟合误差、步长和投影误差推出有限轮联合界，并刻画继续重流有益的预算区及误差地板。至少在一个具体分布族中推出稳定域保持条件，不能把所有轮次共同h>0直接当已知。

**证明入口。** 从加权Helmholtz投影扰动和ODE稳定性入手，追踪端点律改变对下一轮投影度量的影响；用局部邻域不变性或反例判断误差是否被放大。单独e_(k+1)≤ρe_k+δ_k的几何级数不足贡献，难点在数据训练和边际偏移能否满足该递推。

**视觉意义、最小验证。** 区分“路径更直/代价更低”与“图像分布更正确”；先在可解析Gaussian基线和二维非Gaussian混合中扫样本与ODE预算，记录边际W2、运输代价和投影残差，再决定是否做低分辨率图像。Gaussian基线只验证实现，不能充当论文增量。

**近邻与阻断。** v2已覆盖Gaussian矩阵实现、局部projection stability与统计OT估计，须逐条排除；predictive FM的全局拟合接口不能解决本卡优化问题。[Bridge Matching via EM相关原文](https://openreview.net/pdf?id=Bd4DTPzOGO)已讨论非精确更新，尚未完成逐条件核验；另[When and how can inexact generative models still…](https://openreview.net/pdf?id=QBjuYL4gAX)为直接近邻未读全，构成阻断。

**停止条件。** 仅为现成流稳定性界加标准统计误差；只能假设每步边际精确或直接访问真实投影；已有近似bridge/reflow结果覆盖；没有预算阈值或可检验的生成失败预测。当前为次选，未确认开放。

## 淘汰项与覆盖损失

不再立“退火PnP没有非凸收敛”“扩散学习不适应低维”“FM没有minimax率”“普通reflow必达OT”“3DGS好渲染必对应好表面”等宽题。已有明确结果或反例。

本轮使用web发现、arXiv官方API核版本和原文关键节，官方ICLR/AAAI/CVF入口补视觉语境。未做完整COLT/ALT/AISTATS逐会场扫描，也未取得完整前向引文图。标题/别名/主题后续检索仅是定向查重；无召回率或饱和声明。CVPR Findings、workshop与主会分开，未将仅有theoretical字样的工程论文自动收为理论候选。旧A1/T1/S1状态不因本轮未覆盖而改变。

优先完成R1的经典continuation查重和反例，再决定是否投入证明；R2先过近似生成/bridge近邻。15篇为种子数，不代表15篇全文已审完。

## OpenAlex 补充查证（2026-10-28）

已从环境变量 OPENALEX_API_KEY 读取凭据，直接按arXiv DOI查询四篇核心论文：2609.32393、2609.37529、2608.02487、2610.28898，均成功解析OpenAlex记录。逐篇又执行 `cites:W…` 前向查询，当前均返回0条；四篇的 referenced_works 也均为空，无法从该索引展开后向近邻。

这说明本次OpenAlex记录的引用覆盖不足，不支持“没有后续/没有参考文献/缺口仍开放”的结论。它补充了书目身份核对，尚未解除R1/R2的新颖性阻断；后向追溯继续以原文参考文献为准。查询端点、状态、OpenAlex ID和返回计数见 `openalex-check.json`（原记录所指附件，当前工作区未提供），日志不含密钥。主库新增独立openalex_verification字段，未把OpenAlex计数混成Scholar/S2引用数。
