# 退火 RED 的理论补强：算子—后验判据、终端记忆消失与多竞争模板阈值

2026-11-10 · 理论补充修订稿

这份补充针对当前稿件的具体缺口：三模板认证实例已经说明风险分离能够发生，但拟合出的 GMM 与实际观测算子是否进入这一机制，仍缺少可计算的判断方法。下面给出完整的命题和证明，并区分可证明的结论与仍待验证的外推。

最适合进入正文的是命题 1–2：在任意有限维、任意有限模板数和任意线性观测算子下，从后验协方差与实际算子构造终端收缩证书。命题 3 把“有效正则强度变化”和“后验权重变化”分解为两个精确项。命题 4 将原三模板的单步临界质量结果推广到一类多竞争模板配置。

本轮在工作区取得原 patch 的实际算子、GMM 参数和逐样本状态，补充了可复跑的共同终端球检查。主比较继续使用原始 RED 更新。新增结论认证的是导出的有限批状态及其后的精确终端映射，前期舍入和新观测总体风险不在认证范围。

## 1. 模型与记号

设先验为共同各向同性协方差的有限高斯混合：

$$
p_\tau(x)=\sum_{i=0}^{K-1}\pi_i\,\mathcal N(x;\mu_i,\tau^2I_d),
\qquad \pi_i>0,\quad \sum_i\pi_i=1.
$$

观测满足 \(y=Ax^\star+\eta\)，记 \(B=A^\top A\succeq0\)、\(b=A^\top y\)。对总方差 \(v=\tau^2+\sigma^2>\tau^2\)，定义

$$
w_i(x,v)=
\frac{\pi_i\exp[-\|x-\mu_i\|^2/(2v)]}
{\sum_j\pi_j\exp[-\|x-\mu_j\|^2/(2v)]},
\qquad m_v(x)=\sum_iw_i(x,v)\mu_i,
$$

$$
C_v(x)=\sum_iw_i(x,v)(\mu_i-m_v(x))(\mu_i-m_v(x))^\top.
$$

\(C_v\) 是**模板均值的后验协方差**，不是包含分量内部方差的完整 MMSE 后验协方差。原始 RED 更新为

$$
T_v(x)=x-\gamma\{Bx-b+c(v)[x-m_v(x)]\},
\qquad c(v)=\lambda(1-\tau^2/v)>0.
\tag{1}
$$

以下使用欧氏范数；没有假定模板方向是 \(B\) 的特征向量。共同各向同性协方差是实质性假设，不能直接套用于具有不同分量协方差的 GMM。

## 2. 命题 1：任意算子的精确曲率判据

令

$$
\mathcal S=\operatorname{span}\{\mu_i-\mu_0:1\le i<K\},
\qquad r_S=\dim\mathcal S\le K-1.
$$

取 \(\mathcal S\) 的正交基矩阵 \(U\)，其正交补基为 \(V\)。记

$$
C_S(x,v)=U^\top C_v(x)U,
$$

$$
G_c=
U^\top BU+cI_{r_S}
-U^\top BV\,(V^\top BV+cI)^{-1}V^\top BU.
\tag{2}
$$

当 \(\mathcal S=\mathbb R^d\) 时，(2) 的最后一项为零。因为 \(B+cI\succ0\)，其 Schur 补 \(G_c\succ0\)。定义

$$
\rho_v(x)=\frac{c(v)}{v}
\lambda_{\max}\!\left[
G_{c(v)}^{-1/2}C_S(x,v)G_{c(v)}^{-1/2}
\right].
\tag{3}
$$

若所有均值相同，约定 \(\rho_v(x)=0\)。

**命题 1。** 更新 (1) 是如下势函数的梯度下降：

$$
F_v(x)=\tfrac12\|Ax-y\|^2-\lambda(v-\tau^2)\log p_v(x),
$$

其中 \(p_v=\sum_i\pi_i\mathcal N(\mu_i,vI)\)。其 Hessian 满足

$$
H_v(x):=\nabla^2 F_v(x)
=B+c(v)I-\frac{c(v)}{v}C_v(x).
\tag{4}
$$

在每个固定状态 \(x\) 上，

$$
H_v(x)\succ0\quad\Longleftrightarrow\quad\rho_v(x)<1.
\tag{5}
$$

若某凸集 \(\mathcal K\) 上 \(\sup_{x\in\mathcal K}\rho_v(x)\le\bar\rho<1\)，则

$$
mI\preceq H_v(x)\preceq LI,\qquad
m=(1-\bar\rho)(\lambda_{\min}(B)+c(v)),\quad
L=\lambda_{\max}(B)+c(v).
\tag{6}
$$

取 \(0<\gamma\le1/L\)，则 \(T_v\) 在 \(\mathcal K\) 上为 \(q\)-收缩，其中

$$
q=1-\gamma m<1.
\tag{7}
$$

**证明。** 对权重求导得

$$
\nabla_xw_i=\frac{w_i}{v}(\mu_i-m_v),\qquad
D_xm_v=\frac{1}{v}C_v.
$$

又 \(\nabla\log p_v=-(x-m_v)/v\)，故得 (4)。这类条件均值—协方差导数恒等式属于已有理论 [R1]；这里把它用于原始 RED 的具体曲率计算。

由于 \(C_v=UC_SU^\top\)，在正交基 \([U,V]\) 下，\(H_v\) 的右下块为 \(V^\top BV+cI\succ0\)，其 Schur 补恰为 \(G_c-(c/v)C_S\)。正定性的 Schur 补判据给出 (5)。

另外，

$$
U^\top(B+cI)^{-1}U=G_c^{-1},
$$

故 (3) 也等于

$$
\frac{c}{v}\lambda_{\max}\!\left[
(B+cI)^{-1/2}C_v(B+cI)^{-1/2}\right].
\tag{8}
$$

因此 \((c/v)C_v\preceq\bar\rho(B+cI)\)，推出 (6)。在凸集上积分 Jacobian \(DT_v=I-\gamma H_v\)，其谱落在 \([0,1-\gamma m]\)，得到 (7)。证毕。

**这个判据补了什么。** 它同时考虑观测算子的弱方向、这些方向与模板子空间的耦合，以及当前后验是否真的在相关模板间保持不确定性。只看模糊算子的零空间，或只看若干模板差分的观测增益，都不能代替 (3)。

特别地，逐对方向的 Rayleigh 商下界不是整个模板子空间的最小特征值下界。例如

$$
B=\operatorname{diag}(1,0),\qquad
\mu_0=(0,0),\quad\mu_1=(1,0),\quad\mu_2=(2,1).
$$

三条模板差分的平方观测增益分别为 \(1,4/5,1/2\)，但它们张成整个平面，而 \(B\) 在该平面上有零特征值。这并不证明 patch 中存在有关的弱方向；它说明仅凭 28 对方向的最小增益，尚不能严格排除这种方向。

计算上，(3) 的最终特征值问题至多是 \(K-1\) 维。构造 \(G_c^{-1}=U^\top(B+cI)^{-1}U\) 仍需进行与实际算子有关的线性求解，不能把全部计算成本说成仅依赖 \(K\)。

## 3. 命题 2：可直接检验的终端球证书

仅写“终端映射在某区域收缩”仍把关键工作放在假设中。下面用均值、权重、算子和一个候选球给出充分条件。

固定终端 \(v_L>\tau^2\)、\(c_L=c(v_L)\)。给定闭球

$$
\mathcal B=\{x:\|x-\bar x\|\le R\},\qquad R>0,
$$

及一个参考模板 \(j\)。对 \(i\ne j\)，定义

$$
d_{ij}=\mu_i-\mu_j,\qquad
\Delta_{ij}=\|\bar x-\mu_i\|^2-\|\bar x-\mu_j\|^2,
$$

$$
\zeta_{ij}=
\log\frac{\pi_i}{\pi_j}
-\frac{\Delta_{ij}-2R\|d_{ij}\|}{2v_L},
\qquad
\beta_{ij}=\frac{1}{1+\exp(-\zeta_{ij})},
$$

$$
\overline C_{\mathcal B}
=\sum_{i\ne j}\beta_{ij}d_{ij}d_{ij}^\top.
\tag{9}
$$

令

$$
\overline\rho_{\mathcal B}
=\frac{c_L}{v_L}
\lambda_{\max}\!\left[
(B+c_LI)^{-1/2}\overline C_{\mathcal B}
(B+c_LI)^{-1/2}
\right].
\tag{10}
$$

**命题 2。** 假设

$$
\overline\rho_{\mathcal B}<1,\qquad
0<\gamma\le\frac{1}{\lambda_{\max}(B)+c_L},
\tag{11}
$$

并定义

$$
m_{\mathcal B}=(1-\overline\rho_{\mathcal B})(\lambda_{\min}(B)+c_L),
\qquad q_{\mathcal B}=1-\gamma m_{\mathcal B}.
$$

若中心残差满足

$$
\|T_{v_L}(\bar x)-\bar x\|
\le (1-q_{\mathcal B})R,
\tag{12}
$$

则 \(\mathcal B\) 对终端更新前向不变；其中恰有一个终端固定点。任意两个从该球出发、使用同一观测和同一终端映射的轨迹满足

$$
\|x_{H+n}^{(a)}-x_{H+n}^{(b)}\|
\le q_{\mathcal B}^{\,n}\|x_H^{(a)}-x_H^{(b)}\|,
\qquad n\ge0.
\tag{13}
$$

\(H\) 可以取为不同调度均已切换到同一终端更新后的任一共同时间。它们此前的调度可以不同。

**证明。** 球内有

$$
\|x-\mu_i\|^2-\|x-\mu_j\|^2
\ge\Delta_{ij}-2R\|d_{ij}\|.
$$

故 \(w_i/w_j\le e^{\zeta_{ij}}\)，进而

$$
w_i\le\frac{w_i/w_j}{1+w_i/w_j}\le\beta_{ij}.
$$

利用以任意固定均值为中心的二阶矩分解，

$$
C_{v_L}(x)
=\sum_iw_i d_{ij}d_{ij}^\top
-(m_{v_L}-\mu_j)(m_{v_L}-\mu_j)^\top
\preceq\overline C_{\mathcal B}.
$$

由命题 1 得球内 \(q_{\mathcal B}\)-收缩。对任意 \(x\in\mathcal B\)，

$$
\|T_{v_L}(x)-\bar x\|
\le q_{\mathcal B}\|x-\bar x\|
 +\|T_{v_L}(\bar x)-\bar x\|
\le R.
$$

闭球完备，Banach 不动点定理给出唯一性和收敛；对两条轨迹逐次应用收缩，得到 (13)。证毕。

**风险版本。** 若相关随机变量具有有限二阶矩，对随机观测几乎处处都有上述结论，且使用统一 \(q<1\)，令 \(R_a(n)=\mathbb E\|x_{H+n}^{(a)}-x^\star\|^2\)，则

$$
\left|\sqrt{R_a(n)}-\sqrt{R_b(n)}\right|
\le q^n
\sqrt{\mathbb E\|x_H^{(a)}-x_H^{(b)}\|^2}.
\tag{14}
$$

这是 \(L^2\) 范数的反三角不等式与 (13) 的直接结果。若只在好事件 \(E\) 上认证，右侧应改为

$$
q^n\sqrt{\mathbb E[\|x_H^{(a)}-x_H^{(b)}\|^2\mathbf1_E]}
+
\sqrt{\mathbb E[
\|x_{H+n}^{(a)}-x_{H+n}^{(b)}\|^2\mathbf1_{E^c}]}.
$$

因此不能把有限样本上的通过率直接写成无条件总体风险定理。

**有更新误差时。** 如果两条轨迹每一步的加性误差分别不超过 \(\delta_a,\delta_b\)，且仍留在同一球内，则 (13) 的右侧增加

$$
(\delta_a+\delta_b)\frac{1-q_{\mathcal B}^n}{1-q_{\mathcal B}}.
$$

若需同时认证每条含误差轨迹的不变性，(12) 左侧分别加上相应误差上界。浮点舍入或持续随机扰动下，不能无条件主张两条数值轨迹的差必趋于零。

### 3.0 更紧的逐对协方差包络

(9) 删除了围绕参考模板二阶矩中的一个半正定项，在两个分量仍有竞争时可能过松。可改用精确恒等式
\[
 C_v(x)=\sum_{i<j}w_iw_j(\mu_i-\mu_j)(\mu_i-\mu_j)^\top.
\]
对同一候选球，记
\[
 t_{ij}=\log(\pi_i/\pi_j)
 -\frac{\|\bar x-\mu_i\|^2-\|\bar x-\mu_j\|^2}{2v},
 \quad
 L_{ij}=t_{ij}-\frac{R\|\mu_i-\mu_j\|}{v},
 \quad
 U_{ij}=t_{ij}+\frac{R\|\mu_i-\mu_j\|}{v}.
\]
球内 \(\log(w_i/w_j)\in[L_{ij},U_{ij}]\)，因此
\[
 \alpha_i=\left[\sum_k e^{L_{ki}}\right]^{-1}
 \quad\Longrightarrow\quad w_i\le\alpha_i.
\]
又令 \(t_{ij}^\circ\) 为 0 在 \([L_{ij},U_{ij}]\) 上的投影，有
\[
 w_iw_j\le\kappa_{ij}:=
 \min\left\{\alpha_i\alpha_j,\,
 \frac1{(1+e^{t_{ij}^\circ})(1+e^{-t_{ij}^\circ})}\right\}.
\]
第二界来自 \(w_i+w_j\le1\)，且给定两者赔率时乘积在其总和为 1 时最大。对数赔率离 0 越近，该最大乘积越大。故
\[
 \overline C_{\mathcal B}^{\rm pair}
 =\sum_{i<j}\kappa_{ij}(\mu_i-\mu_j)(\mu_i-\mu_j)^\top
\]
也是有效的全球 Loewner 上界，可直接替代 (9)。可以在两个有效包络中选择使 (10) 更小的一项，不能逐矩阵元素取最小。该强化保留分量间竞争对乘积的限制，仍无需观测真值或标签。

### 3.1 无需局部球的保守充分条件

设模板直径 \(D_\mu=\max_{i,j}\|\mu_i-\mu_j\|\)。对任意后验权重，

$$
C_v(x)\preceq\frac{D_\mu^2}{4}P_{\mathcal S}.
\tag{15}
$$

证明是对每个方向使用有界随机变量方差不超过取值范围平方四分之一，再注意均值差均位于 \(\mathcal S\)。

因此，若 \(r_S>0\) 且

$$
\frac{c_LD_\mu^2}{4v_L}
<\lambda_{\min}(G_{c_L}),
\tag{16}
$$

则终端映射在适当步长下全局收缩，所有初始化均收敛至同一固定点。所有均值相同时，该结论直接由 \(B+c_LI\succ0\) 得到。(16) 不成立仅表示这个充分条件无结论，不能据此判定存在多个吸引域。

更直接的全局检验可先按实际算子白化。记 \(M=B+c_LI\)，定义
\[
 D_M^2=\max_{i,j}(\mu_i-\mu_j)^\top M^{-1}(\mu_i-\mu_j).
\]
将同一个方向方差界用于白化模板，得到
\[
 \rho_{v_L}(x)\le\frac{c_LD_M^2}{4v_L}\quad\text{对所有 }x.
\]
所以 \(c_LD_M^2/(4v_L)<1\) 也是全局充分条件，通常比 (16) 更紧。它不能反向用于证明多个吸引域。

### 3.2 一个非空性实例

取二维均值 \((-2,0),(0,0),(2,0)\)，等权；
\(B=\operatorname{diag}(0.05,1)\)、\(b=0\)、
\(v_L=0.1\)、\(c_L=0.2\)、\(\gamma=1/4\)。
例如 \(\tau^2=0.01,\lambda=2/9\) 与这些 \(v_L,c_L\) 相容。

以 \(\bar x=0\)、\(R=0.25\)、中间均值为参考，得到

$$
\beta=\operatorname{logistic}(-15),\qquad
\overline\rho_{\mathcal B}=64\beta
\simeq1.9577743\times10^{-5},
$$

$$
q_{\mathcal B}\simeq0.9375012,\qquad
T_{v_L}(0)=0.
$$

故整个球满足命题 2。这只是证书的明确实例，不是对照片实验的代入结果。

### 3.3 现有 patch 的实际代入与向外舍入验证

原始模板子空间维数为 7，\(U^\top BU\) 的最小特征值约 .02509617，低于逐对模板最小增益 .22752。这直接纠正了仅凭逐对增益排除子空间弱方向的解释。白化直径的全局上界约为 1.21022，原 (16) 对应上界约为 196.844，均未通过。全局收缩仍无结论。

对每个观测，从原三条调度在共同时间 \(H=154,1000,5000\) 的输出构造球。中心由终端 5000 步输出再做 10 次 Newton 迭代得到，仅作为证书候选。半径取覆盖三条输出所需最大距离的 1.01 倍，并设 \(10^{-9}\) 的下限。全部选择只使用观测、模型和状态，不用真值 patch。

| 全球检查 | H=154 | H=1000 | H=5000 |
| --- | ---: | ---: | ---: |
| 原参考模板包络，尝试所有参考模板 | 255/256 | 255/256 | 256/256 |
| 加入逐对协方差包络 | 256/256 | 256/256 | 256/256 |

原界失败的都是样本 42，表示包络不足，不是证伪局部收缩。H=154 的逐对界最大 \(\bar\rho\) 约 .441213。进一步统一选取有理上界 \(\rho_*=.45\)，对每个候选球以 60 位向外舍入计算协方差包络，直接用区间 LDL 验证
\[
 \rho_*(B+c_LI)-(c_L/v_L)\overline C_{\mathcal B}^{\rm pair}\succ0.
\]
同时验证 (12)、三条输出都在球内以及步长条件，256 项全部通过。模糊算子具有精确有理 Gram 分解且范数为 1，所以直接取
\[
 m=.55c_L,\qquad
 q=1-\gamma(.55c_L)<.999429.
\tag{17}
\]
于是对该批每个观测、任意两种导出输出，往后的精确终端映射满足 (13)。统一 \(q\) 给出从 154 到 5000 步的距离衰减上界约 .06274。已观察到的均方输出差可进一步衰减，仍不能据此认定共同固定点具有低恢复风险。

这里将导出的 binary64 模型参数、有效系数和状态解释为精确输入。认证没有包络此前 154 步的浮点传播，也没有认证未来浮点计算在无限步下完全消除差异。持续误差应使用本节误差版本。有限批逐观测证书不自动扩展为新观测上的总体风险结论。

[候选球与浮点检查](experiments/results/patch_terminal_pairwise/checks.json) · [向外舍入证书](experiments/results/patch_terminal_interval/certificate.json) · [验证器](experiments/code/patch_terminal_interval.py)

后续冻结模型的独立验证又加入原flower的新观测噪声和3张新增Set12照片，各256个patch、3个噪声种子，共3072个观测。全部导出状态通过区间共同球证书，其中3062个仍可用原 \(\rho=.45\)，其余10个使用最高 \(\rho=.798\) 的样本特定上界，统一 \(q<.999791\)。中间调度的154步恢复误差在两张图改善、两张图变差，没有跨图一致的优势。因而独立证据支持终端历史消失，正向真实patch风险分离仍未建立。[独立验证设置与限制](experiments/results/patch_independent_validation/analysis.md)


## 4. 命题 3：尺度变化的两项精确分解

为分析归因，暂将 \(c\) 与 \(v\) 当作两个独立参数：

$$
T_{c,v}(x)=x-\gamma[Bx-b+c(x-m_v(x))].
$$

这是一种分析记号。原方法仍限制在曲线 \(c=\lambda(1-\tau^2/v)\) 上。

**命题 3。** 对任意固定状态 \(x\)，

$$
T_{c,v}(x)-T_{\tilde c,\tilde v}(x)
=-\gamma(c-\tilde c)[x-m_v(x)]
+\gamma\tilde c[m_v(x)-m_{\tilde v}(x)].
\tag{18}
$$

第一项为系数变化，第二项为后验均值变化。设 \(z=1/v\)，
\(E_i(x)=\|x-\mu_i\|^2/2\)，则

$$
\frac{\partial m_{1/z}(x)}{\partial z}
=-\operatorname{Cov}_{w(x,1/z)}(\mu,E).
\tag{19}
$$

因此

$$
\|m_v(x)-m_{\tilde v}(x)\|
\le |v^{-1}-\tilde v^{-1}|
\sup_{z\in J}\|\operatorname{Cov}_{w(x,1/z)}(\mu,E)\|,
\tag{20}
$$

其中 \(J\) 为两个逆方差间的闭区间。此外，

$$
\|\operatorname{Cov}_w(\mu,E)\|
\le\sqrt{\lambda_{\max}(C_w)\operatorname{Var}_w(E)}.
\tag{21}
$$

**证明。** (18) 由加减同一项得到。归一化指数权重满足
\(\partial_zw_i=-w_i(E_i-\sum_\ell w_\ell E_\ell)\)，求和得到 (19)，积分得到 (20)。对任意单位向量 \(u\)，向量协方差的投影为
\(\operatorname{Cov}(u^\top\mu,E)\)；Cauchy–Schwarz 不等式给出 (21)。证毕。

**轨迹版本。** 若两条轨迹均留在区域 \(\mathcal K\)，第 \(k\) 个基准映射 \(T_{c_k,v_k}\) 在该区域内为 \(q_k\)-Lipschitz，记

$$
R_k=\sup_{x\in\mathcal K}\|x-m_{v_k}(x)\|,
$$

$$
M_k=\sup_{x\in\mathcal K,z\in J_k}
\|\operatorname{Cov}_{w(x,1/z)}(\mu,E(x))\|.
$$

则两条输出的距离 \(e_k\) 满足

$$
e_{k+1}\le q_ke_k+\gamma\left[
|c_k-\tilde c_k|R_k
+\tilde c_k|v_k^{-1}-\tilde v_k^{-1}|M_k
\right].
\tag{22}
$$

反复代入得到

$$
e_n\le
\left(\prod_{k=0}^{n-1}q_k\right)e_0
+
\gamma\sum_{\ell=0}^{n-1}
\left(\prod_{k=\ell+1}^{n-1}q_k\right)
\left[
|c_\ell-\tilde c_\ell|R_\ell
+\tilde c_\ell|v_\ell^{-1}-\tilde v_\ell^{-1}|M_\ell
\right].
\tag{23}
$$

空乘积按 1 计。(22) 通过先比较同一映射下的两个状态，再在第二个状态使用 (18)–(20) 得到。无需假定这两项在方向上相同。

这为固定 \(c\) 与固定后验权重的归因对照提供了明确的理论量。如果后验集中使 (21) 很小，改变平滑后的权重对更新影响就小，而改变 \(c\) 仍可能影响轨迹。但两项之间可能抵消，(23) 是上界，不能由单一输出差唯一反演各自贡献。

### 4.1 四角对照与交互项

(18) 的分解依赖于加减项的顺序。为直接对应固定 \(c\) 与固定后验权重的实验，令 \(m_0=m_{v_0}(x)\)、\(m_1=m_{v_1}(x)\)、\(\Delta c=c_1-c_0\)，定义同一状态上的两项单独改动

$$
A_c=T_{c_1,v_0}(x)-T_{c_0,v_0}(x)
=-\gamma\Delta c(x-m_0),
$$

$$
A_v=T_{c_0,v_1}(x)-T_{c_0,v_0}(x)
=\gamma c_0(m_1-m_0).
$$

四角混合差为

$$
\begin{aligned}
I={}&T_{c_1,v_1}(x)-T_{c_1,v_0}(x)\\
&-T_{c_0,v_1}(x)+T_{c_0,v_0}(x)
=\gamma\Delta c(m_1-m_0).
\end{aligned}
\tag{23a}
$$

因此总更新差精确等于 \(A_c+A_v+I\)，且

$$
\|I\|\le\gamma|\Delta c|\,|v_1^{-1}-v_0^{-1}|
\sup_{z\in J}\|\operatorname{Cov}_{w(x,1/z)}(\mu,E)\|.
\tag{23b}
$$

(18) 将 \(I\) 合入系数项，反向加减则会将它合入平滑项。各自运行的归因轨迹还会经过不同状态，不能将最终输出差直接相加；(22)–(23) 控制这种状态传播。轨迹上界要求 \(c_k,\tilde c_k\ge0\)、\(v_k,\tilde v_k>0\)、\(q_k\ge0\)，并在需要有限上界时要求 \(R_k,M_k<\infty\)。原 RED 的正系数、总方差限制满足这些参数要求。

## 5. 命题 4：多竞争模板的精确临界质量

这部分推广当前三模板的固定状态结果。它适用于多个模板都投影在参考模板后方的结构化配置，不覆盖任意排列的所有 GMM。

取 \(\mu_0=0,\mu_1=e,\|e\|=1\)，固定状态 \(x\)，令

$$
s=e^\top x\in(0,1/2),\qquad P=\tfrac12-s>0.
$$

另有 \(M\ge1\) 个竞争模板 \(\nu_j\)，满足

$$
e^\top\nu_j=-a_j,\qquad a_j>0,
$$

$$
Q_j=\tfrac12(\|x-\nu_j\|^2-\|x\|^2)>P.
\tag{24}
$$

先验质量比取为

$$
\pi_0:\pi_1:\pi_{\nu_1}:\cdots:\pi_{\nu_M}
=1:r:hq_1:\cdots:hq_M,\qquad q_j>0,\quad\sum_jq_j=1.
$$

\(h\) 是竞争模板的总质量比，\(q_j\) 固定其内部组成。定义

$$
\bar a=\sum_jq_ja_j,\qquad
\bar Q=\sum_jq_jQ_j,\qquad
\overline{aQ}=\sum_jq_ja_jQ_j,
$$

$$
C_s=(1-s)r-s>0,\qquad
h_{\max}=\frac{C_s}{\bar a+s},\qquad h_0=\frac r{\bar a}.
\tag{25}
$$

设 \(z_L=1/v_L\)、\(\ell(z)=1-\tau^2z\)，并假定

$$
v_L>\tau^2,\qquad
re^{-Pz_L}\le\frac{s}{1-s}.
\tag{26}
$$

水平后验均值和归一化先验漂移为

$$
m_h(z)=
\frac{re^{-Pz}-h\sum_j a_jq_je^{-Q_jz}}
{1+re^{-Pz}+h\sum_jq_je^{-Q_jz}},
\qquad
g_h(z)=\ell(z)[m_h(z)-s].
\tag{27}
$$

对固定状态，实际 RED 水平位移除以 \(\gamma\) 为
\(e^\top(b-Bx)+\lambda g_h(z)\)。故比较 \(g_h\) 就是比较实际单步位移，不要求 \(e\) 与 \(B\) 对齐。

定义二次多项式

$$
\begin{aligned}
D(h)={}&-Pr\\
&+h\left[(1+r)\overline{aQ}+r\bar Q-Pr(1+\bar a)\right]\\
&+h^2\left(\overline{aQ}-\bar a\bar Q\right)\\
&-\tau^2\left[C_s-h(\bar a+s)\right](1+r+h).
\end{aligned}
\tag{28}
$$

**命题 4。** 在 (24)–(26) 下，对 \(0<h\le h_{\max}\)，高方差极限端点 \(z=0\) 的单步位移不小于终端 \(z=z_L\)。存在某个有限中间方差 \(v\in(v_L,\infty)\) 严格超过两端位移，当且仅当

$$
D(h)>0.
\tag{29}
$$

多项式 \(D\) 在 \((0,h_0)\) 内恰有一个根 \(h_{\rm crit}\)，且

$$
0<h_{\rm crit}<h_0,\qquad
0<h\le h_{\max}\Longrightarrow
\big[D(h)>0\Longleftrightarrow h>h_{\rm crit}\big].
\tag{30}
$$

若 \(h_{\rm crit}\ge h_{\max}\)，此可比较质量区间内没有严格改进。固定其他标量输入，增加任一 \(Q_j\) 会严格降低 \(h_{\rm crit}\)，只要 (24) 保持成立。

注意：这里的唯一性限定在 \((0,h_0)\)。多个竞争模板引入了
\(\overline{aQ}-\bar a\bar Q=\operatorname{Cov}_q(a,Q)\)，它可以为负；不能一概断言 \(D\) 在整个正半轴只有一个根。

### 5.1 证明：端点比较与凸判据

由 \(h\le h_{\max}\)，

$$
g_h(0)=\frac{C_s-h(\bar a+s)}{1+r+h}\ge0.
$$

由 (26) 的指数不等式，删除竞争模板后终端的目标责任度已不超过 \(s\)；加入投影为负的竞争模板只会进一步降低水平均值。因此 \(g_h(z_L)\le0\)，高方差端点较优。

置 \(u=g_h(0)\ge0\)，并记

$$
A_u(z)=(1-s)\ell(z)-u,\qquad
U_u(z)=\frac{s\ell(z)+u}{A_u(z)},\qquad
W_{u,j}(z)=\frac{(a_j+s)\ell(z)+u}{A_u(z)}.
$$

有 \(0\le u\le C_s/(1+r)<1-s\)，所以 \(A_u(0)>0\)。记

$$
I_u=\{z\in[0,z_L]:A_u(z)>0\}.
$$

由于 \(A_u\) 随 \(z\) 不增，\(I_u\) 是包含零的初始区间。仅在 \(I_u\) 上进行以下除法，代数整理给出严格等价式

$$
g_h(z)>u
\quad\Longleftrightarrow\quad
J_u(z):=
U_u(z)e^{Pz}
+h\sum_jq_jW_{u,j}(z)e^{-(Q_j-P)z}<r.
\tag{31}
$$

并且 \(J_u(0)=r\)。为证明凸性，写成

$$
U_u=\frac{s}{1-s}+\frac{u}{(1-s)A_u},\qquad
W_{u,j}=\frac{a_j+s}{1-s}
+\frac{(1+a_j)u}{(1-s)A_u}.
$$

常数函数与正仿射函数倒数均为对数凸函数，非负加权和保持对数凸。因此 \(U_u,W_{u,j}\) 均对数凸，乘指数后仍凸。其中 \(U_ue^{Pz}\) 严格凸：\(P>0\)、\(U_u>0\)，且 \(U_u',U_u''\ge0\)。故 \(J_u\) 严格凸。

对区间外的 \(A_u(z)\le0\)，所有权重均为正且模板水平投影至多为 1，故 \(m_h(z)<1\)。由于 \(\ell(z)>0\)，

$$
g_h(z)-u<(1-s)\ell(z)-u=A_u(z)\le0.
$$

这些状态不可能严格改进。因此存在 \(z\in(0,z_L)\) 使 \(g_h(z)>u\)，当且仅当 \(I_u\cap(0,z_L)\) 上有 \(J_u(z)<r\)，由严格凸性又等价于 \(J_u'(0)<0\)。从 (31) 交叉相乘可进一步写成

$$
g_h(z)-u
=\frac{A_u(z)e^{-Pz}}{1+re^{-Pz}+h\sum_jq_je^{-Q_jz}}
\,[r-J_u(z)].
$$

在 \(z=0\) 处求导，前因子严格为正，故
\(\operatorname{sign}g_h'(0)=-\operatorname{sign}J_u'(0)\)。
对 (27) 直接求导得

$$
g_h'(0)=\frac{D(h)}{(1+r+h)^2}.
\tag{32}
$$

从而得到 (29)。

### 5.2 证明：阈值唯一性及距离单调性

显然 \(D(0)=-Pr-\tau^2C_s(1+r)<0\)。
在 \(h=h_0=r/\bar a\) 时，\(m_h(0)=0\)，且

$$
m_h'(0)=
\frac{r(\overline{aQ}/\bar a-P)}{1+r+h_0}>0,
$$

因为 \(\overline{aQ}/\bar a\) 是各个 \(Q_j>P\) 的正加权平均。因此
\(g_h'(0)=m_h'(0)+\tau^2s>0\)，即 \(D(h_0)>0\)。

\(D\) 的次数至多为二，两个端点异号，故在 \((0,h_0)\) 内恰有一个根，并在该根处从负变正、满足 \(\partial_hD(h_{\rm crit})>0\)。又 \(h_{\max}<h_0\)，得到 (30)。

对任意竞争模板，

$$
\frac{\partial D(h)}{\partial Q_j}
=hq_j\left[a_j(1+r+h)+(r-h\bar a)\right]>0,
\qquad 0<h<h_0.
\tag{33}
$$

由隐函数定理，

$$
\frac{\partial h_{\rm crit}}{\partial Q_j}
=-\frac{\partial_{Q_j}D(h_{\rm crit})}
{\partial_hD(h_{\rm crit})}<0.
$$

这里是保持其他标量参数不变的局部比较；同时改变模板位置、投影和距离差时，不能只引用此偏导判断总变化。证毕。

### 5.3 与原三模板结果、完整风险定理的关系

当只有一个竞争模板时，
\(\bar a=a,\bar Q=Q,\overline{aQ}=aQ\)，协方差项消失，(28) 化为

$$
D(h)=-Pr+h[aQ+(1+a)(Q-P)r]
-\tau^2[C_s-(a+s)h](1+r+h),
$$

恰为当前稿件的三模板多项式。

命题 4 证明的是固定状态处的严格单步端点改进。即使条件成立，也还需要共同障碍、沿通道的均匀进展、横向不变性及正确终端区域入口，才能调用原稿的一般风险转移定理。它不把“局部有利方差存在”自动升级为“从给定初始化可以恢复”。

高方差端点在这里是 \(v=\infty\) 的极限。若取得严格改进，其正间隔由连续性保证：足够大的有限高方差端点也被同一个中间步严格超过。此处并未证明多竞争模板在任意给定有限端点区间上都有同一质量公式。

### 5.4 限定根区间与放松分母条件的实例

双正根可以由实际模板几何实现。取 \(x=(.3,3)\)、\(e=(1,0)\)、\(r=1.29\)、\(\tau^2=10^{-4}\)、\(v_L=.01\)，两个竞争模板为

$$
\nu_1=(-.1,3+\sqrt{28.93}),\qquad
\nu_2=(-3,3+\sqrt{2.2}),\qquad q_1=q_2=1/2.
$$

于是 \(a=(.1,3)\)、\(Q=(10,2)\)，且

$$
D(h)=-5.799815h^2+15.09746335h-.258138087.
$$

其两个正根约为 \(.01721192\) 和 \(2.58588204\)，而 \(h_0\simeq.83225806\)、\(h_{\max}\simeq.32594595\)。只有第一根位于命题的阈值区间。第二根不能用来推断可比较质量区间之外的端点优势。

证明还消除了仅为全区间分母为正而添加的限制。例如 \(s=.3,r=10,a=1,Q=2.82,\tau^2=.009,v_L=.01,h=.1\) 满足现有条件，但 \(C_s/(1+r)\simeq.60909\) 大于 \((1-s)\ell(z_L)=.07\)。取 \(x=(.3,-.01)\)、竞争模板 \((-1,2)\) 即可实现该 \(Q\)。此时 \(D(h)=2.865657>0\)，在 \(v\simeq3.78556\) 处有 \(g_h\simeq.59462710>g_h(0)\simeq.59189189\)。只在 \(A_u>0\) 的初始区间使用凸判据，足以覆盖这一实例。

## 6. 如何放入现稿

建议保留原风险分离定理的主线，同时增加两种有明确作用的结果。

| 结果 | 建议位置 | 支撑的主张 | 不能替代的结论 |
|---|---|---|---|
| 命题 1–2 | 正文新增一个一般 GMM 小节；球证书证明放补充稿 | 可由实际算子和拟合先验检查调度历史是否会被终端消除 | 新观测总体风险；共同固定点准确 |
| 命题 3 | patch 归因实验附近简述，完整式放补充稿 | 系数变化与后验变化有精确代数分解和累计误差界 | 两项各自贡献的唯一统计识别 |
| 命题 4 | 替换或接续原三模板临界质量命题 | 竞争机制不限于一个第三模板 | 任意多模板配置的完整风险分离 |
| 原障碍—通道—终端认证 | 继续作为正向风险保证 | 结构化参数族中确有非消失风险与一致恢复的分离 | 真实图像自动满足同样条件 |

正文可以形成更准确的贡献叙述：**给出风险分离的认证实例，并在一般有限 GMM 中提供排除持久终端调度差异的可计算证书；进一步刻画结构化多竞争模板中单步尺度优势的质量阈值。**

这会实质增强适用范围和可检验性，但仍需避免将 Schur 补、条件协方差导数或收缩定理本身作为全新数学工具来宣传。它们是标准工具；本补强的价值在于将其组织为与实际算子、GMM 参数和终端区域直接对应的证书，以及给出多竞争模板的具体阈值扩展。

本轮实际局部证书已通过，因而将该批实验从“没有观察到优势”补强为“导出状态所处共同球内，精确终端映射的持久调度差异被排除”。这仍不保证低恢复误差或总体风险。未通过全局充分界的结论是未知，不能反向断言中间带一定有优势。

## 7. 相关文献应补在具体论证处

以下来源已核对相应论文或作者机构页面。R1 和 R3 尤其适合与本次新增理论一起补入；R2、R4 如果当前参考文献已有，则应加强正文对照而不重复计数。

- **[R1]** Alex Dytso, H. Vincent Poor, Shlomo Shamai Shitz. *Conditional Mean Estimation in Gaussian Noise: A Meta Derivative Identity With Applications*. IEEE Transactions on Information Theory, 69(3):1883–1898, 2023. [DOI](https://doi.org/10.1109/TIT.2022.3216012)。用于说明条件均值 Jacobian 与条件协方差恒等式的已有基础；本文需把创新落在具体证书和调度结论上。
- **[R2]** Miguel A. Carreira-Perpiñán. *Gaussian mean-shift is an EM algorithm*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 29(5):767–776, 2007. [作者稿](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/pami07.pdf)。用于对照高斯混合固定点迭代、局部收敛速率和吸引域分析；不能把“混合模型存在不同吸引域”作为本稿独有贡献。
- **[R3]** Arghya Sinha, Bhartendu Kumar, Chirayu D. Athalye, Kunal N. Chaudhury. *Linear Convergence of Plug-and-Play Algorithms with Kernel Denoisers*. IEEE Transactions on Signal Processing, 73:2646–2659, 2025. [DOI](https://doi.org/10.1109/TSP.2025.3573044) · [全文](https://arxiv.org/html/2505.15318v1)。其核去噪器是固定线性算子。对称情形分析普通 PnP，非对称情形需相应缩放算法及加权范数。本补充分析状态相关的 GMM 后验协方差与残差相加的 RED 更新。
- **[R4]** Edward T. Reehorst, Philip Schniter. *Regularization by Denoising: Clarifications and New Interpretations*. [论文](https://arxiv.org/abs/1806.02296)。用于交代 RED、score 与 Jacobian 对称性的边界。本补充的势函数来自精确 GMM score，不是对任意实际去噪器都宣称存在同一势函数。
- **[R5，已对照主要定理]** Tobias Wolf, Jalal Fadili, Jin Guo, Roy Y. He. *Physics Matters in PnP: Recovery Guarantees with the MMSE and NN Denoisers*. arXiv:2607.29525v1, 2026. [原始全文](https://arxiv.org/html/2607.29525v1)。其 Eq. (6)/(16) 为 \(x^+=D(x+\gamma P(y-Ax))\)，文中的反投影算子命名与本稿 \(B=A^\top A\) 不同。Theorem 3.2、3.6 分别给点态和 Wasserstein 恢复界，要求紧支撑、支撑位于去噪噪声协方差的像空间、\(PA\) 对称半正定及相关切锥上的单射等条件。本稿未截断 GMM 和残差相加的 RED 不可直接套用这些结论。其算子与先验联合分析应作为定位对照。

这一组是有针对性的定位补充，不是穷尽性新颖性检索。特别是命题 1–2 的标准工具来源应坦诚引用。

## 8. 本次核验及其限度

原补稿报告了以下一致性检查。本工作区未收到其原始检查脚本，故保留为来源说明，不将原报告计数当作本轮独立复核：

| 检查 | 结果 |
|---|---|
| 多竞争模板 \(g_h'(0)\) 与二次式 (28) | 符号化简恒等相等 |
| 单竞争模板退化到现稿公式 | 符号化简恒等相等 |
| 1,500 组质量—几何组合的二次式与凸最小值判据 | 全部一致 |
| 200 组非对齐算子的低维 Schur 判据与全维判据 | 最大绝对差 \(2.45\times10^{-14}\) 以内 |
| 1,000 个球内抽样点的协方差矩阵上界 | 未发现违反 |
| 100 组温度导数及两项分解 | 分解误差小于 \(10^{-12}\)；导数差分误差小于 \(6\times10^{-10}\) |

随机检查只用于发现公式或实现错误，不代替全区域证明，也不是对照片实验的复现。球证书的全区域有效性来自命题 2 的解析距离间隔界。

本轮可复跑核验另见 [threshold_checks.json](experiments/results/theory_extension_review/threshold_checks.json) 和 [检查脚本](experiments/code/theory_extension_checks.py)。包括 128 组阈值检查（其中 69 组不满足已删去的多余分母条件）、64 组四角恒等式、双正根实例和单竞争退化。实际 patch 另有上述全球包络与区间 LDL 证书。球内结论来自解析包络，未以抽样点替代全区域证明。

## 附录 A：终端球证书的数值检查函数

以下 NumPy 函数保留原参考模板版本，用于把命题 2 代入现有 patch 参数。实际工作区的 [terminal_ball.py](experiments/code/terminal_ball.py) 还实现了 §3.0 的逐对强化版本。输入 \(B=A^\top A\)，不是 \(A\)；v 是总方差；c 应由原实验的 \(\lambda,\tau,v\) 计算。points 可传同一观测下三种调度在共同时间 \(H\) 的输出。

它使用普通浮点数，返回的是带明确裕度的数值检查。若要称为包含舍入误差的严格机器证书，需对特征值、残差及比较另做区间包络，尤其不能把接近零的裕度当作可靠通过。

~~~python
import numpy as np

def terminal_ball_check(B, b, means, pi, v, c, gamma,
                        center, radius, reference, points=None):
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    means = np.asarray(means, dtype=float)
    pi = np.asarray(pi, dtype=float)
    center = np.asarray(center, dtype=float)
    d = len(center)
    if B.shape != (d, d) or b.shape != (d,):
        raise ValueError("B and b have incompatible dimensions")
    if means.ndim != 2 or means.shape[1] != d:
        raise ValueError("means must have shape (K, d)")
    if pi.shape != (len(means),) or np.any(pi <= 0):
        raise ValueError("pi must contain positive mixture weights")
    if not (v > 0 and c > 0 and gamma > 0 and radius > 0):
        raise ValueError("v, c, gamma, radius must be positive")
    if not 0 <= reference < len(means):
        raise ValueError("reference is out of range")
    if not np.allclose(B, B.T, atol=1e-12, rtol=1e-12):
        raise ValueError("B must be symmetric")
    B = (B + B.T) / 2
    eigen_B = np.linalg.eigvalsh(B)
    if eigen_B[0] < -1e-12:
        raise ValueError("B must be positive semidefinite")

    delta = means - means[reference]
    energy = np.sum((center - means)**2, axis=1)
    gaps = energy - energy[reference]
    log_odds = (np.log(pi) - np.log(pi[reference])
                - (gaps - 2*radius*np.linalg.norm(delta, axis=1))/(2*v))
    beta = np.exp(-np.logaddexp(0, -log_odds))
    beta[reference] = 0
    Cbar = (delta.T * beta) @ delta

    chol = np.linalg.cholesky(B + c*np.eye(d))
    left = np.linalg.solve(chol, Cbar)
    normalized = np.linalg.solve(chol, left.T).T
    normalized = (normalized + normalized.T) / 2
    rho = (c/v)*np.linalg.eigvalsh(normalized)[-1]
    lower_curvature = (1-rho)*(eigen_B[0]+c)
    upper_curvature = eigen_B[-1]+c
    q = 1-gamma*lower_curvature

    log_w = np.log(pi) - energy/(2*v)
    weights = np.exp(log_w-log_w.max())
    weights /= weights.sum()
    m = weights @ means
    residual = gamma*np.linalg.norm(B@center-b+c*(center-m))
    containment_margin = (1-q)*radius-residual
    starts_inside = True
    max_start_distance = None
    if points is not None:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != d:
            raise ValueError("points must have shape (N, d)")
        max_start_distance = float(np.linalg.norm(points-center, axis=1).max())
        starts_inside = max_start_distance <= radius
    passed = (rho < 1 and gamma*upper_curvature <= 1
              and containment_margin >= 0 and starts_inside)
    return dict(
        passed=bool(passed), rho_upper=float(rho), q_upper=float(q),
        curvature_margin=float(1-rho),
        step_margin=float(1-gamma*upper_curvature),
        containment_margin=float(containment_margin),
        residual=float(residual), starts_inside=bool(starts_inside),
        max_start_distance=max_start_distance,
        note="Floating-point diagnostic; not a rounded interval certificate."
    )
~~~

如果函数不通过，应保留具体失败项：曲率界太宽、步长不满足充分条件、中心残差太大，或候选球没有覆盖所有输出。增大半径会改善覆盖、却可能恶化后验协方差界；不应把调参后通过的球等同于总体风险保证。
