# 标准网络整图去模糊：准备与运行

本入口检验固定 RED 更新下的经验尺度历史效应。代码只读取本地模型和图像；缺文件直接报错，所有下载命令单独放在本页，由用户执行。当前进展与是否启动实验见根目录 [status.md](../status.md)。

## 环境与本地资源

以下 PowerShell 命令均从仓库根目录运行。本机 Python 为 `H:/anaconda3/envs/py312/python.exe`；已有 PyTorch 2.9.1+cu126、CUDA 可见。依赖安装按需执行，不需要重新训练网络。

```powershell
& H:/anaconda3/envs/py312/python.exe -m pip install -r experiments/requirements_network.txt
New-Item -ItemType Directory -Force experiments/assets/weights, experiments/assets/kernels, experiments/assets/data | Out-Null
```

资源统一放在被 Git 忽略的 `experiments/assets/`，不要放进源码或提交权重。Hugging Face 仓库、文件名和下面的版本于 2026-11-16 核对；版本号用于指定下载版本，不做默认源码/数据哈希校验。

### 权重：优先 Hugging Face

```powershell
& H:/anaconda3/envs/py312/Scripts/hf.exe download deepinv/drunet drunet_gray.pth --revision 7e079a6800958ae777b48e41fc1202a2ee21ecbe --local-dir experiments/assets/weights
& H:/anaconda3/envs/py312/Scripts/hf.exe download deepinv/FFDNet ffdnet_gray.pth --revision bdea5b3a7f232f80a9292fda4009693edf30e27a --local-dir experiments/assets/weights
```

来源：[DRUNet 文件](https://huggingface.co/deepinv/drunet/blob/7e079a6800958ae777b48e41fc1202a2ee21ecbe/drunet_gray.pth)、[FFDNet 文件与转换说明](https://huggingface.co/deepinv/FFDNet/tree/bdea5b3a7f232f80a9292fda4009693edf30e27a)、[HF CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)。DRUNet 选原版 `drunet_gray.pth`，不要替换成名称带 `deepinv` 或 `finetune` 的其他模型。

HF 的 FFDNet 是 DeepInv 对 KAIR 权重的转换版：参数名为 `blocks.*`，第一层接收像素重排后的图像与噪声图，共 8 通道。原 KAIR 版参数名为 `model.*`，先重排图像再拼接噪声，共 5 通道。代码显式支持两种布局、严格加载，并在校准/评价输出记录布局；不把两种布局的张量直接混装。使用作者原版 `ffdnet_gray.pth` 时只需更改本地权重路径。

网络结构来源与许可见 [NOTICE](code/real_network/NOTICE.md)。网络灰度、整图推理，不使用 x8、切块、混合精度、去噪后的 uint8 转换或迭代内裁剪。DRUNet 固定在底部/右侧 replicate pad 到 8 的倍数，FFDNet 到 2 的倍数，随后裁回原大小；每次逻辑去噪恰好一次实际前向。

### 模糊核：Hugging Face

```powershell
& H:/anaconda3/envs/py312/Scripts/hf.exe download deepinv/degradations Levin09.npy --repo-type dataset --revision c3e9c9d87bdd246675149d7ed790eaabbb24191d --local-dir experiments/assets/kernels
```

[核文件来源](https://huggingface.co/datasets/deepinv/degradations/tree/c3e9c9d87bdd246675149d7ed790eaabbb24191d)。示例取数组第一个核，另有本地生成的 25×25、标准差 1.6 高斯核。核按保存方向使用、归一化为和 1；不按恢复结果挑核。Levin09 本地文件可以是可变大小核的 NumPy 对象数组；仅从上述已知来源手动取得文件。冻结计划保存实际 2D 数组，评价不重新读取或选择核。也支持本地 `.mat`（`kernels` 字段、零起点索引）或二维 `.npy`。

### 灰度 BSD68 / Set12：作者仓库

已核对的 HF `deepinv/CBSD68` 和 `deepinv/images/CBSD68.zip` 是彩色集合，本入口不把其灰度转换标作作者的标准灰度 BSD68。本轮使用 [DnCNN 作者提供的 BSD68](https://github.com/cszn/DnCNN/tree/master/testsets/BSD68) 和 [Set12](https://github.com/cszn/DnCNN/tree/master/testsets/Set12)，只下载需要的目录：

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/cszn/DnCNN.git experiments/assets/dncnn
git -C experiments/assets/dncnn sparse-checkout set testsets/BSD68 testsets/Set12
```

目录分别为 `experiments/assets/dncnn/testsets/BSD68`（68 图）和 `experiments/assets/dncnn/testsets/Set12`（12 图）。已有本地同版本数据可直接使用，无需重复下载。标准集合标签要求读入原始 8 位灰度图且数量完整；子集使用 `--dataset custom`。Set12 的 01、02、10 在旧 patch 工作中已经暴露，评价 CSV 会标记，不能称新盲测。

开发图像另放 `experiments/assets/data/dev`，建议固定 12–16 张灰度整图，并自行保存来源与选择说明；不要把 BSD68/Set12 放进去选参。DRUNet 训练包含 BSD400、Waterloo、DIV2K 的 900 张和 Flickr2K，普通 DIV2K validation 不能直接视作其未见数据。测试集与本轮开发选参的独立性、与预训练的关系分别交代。代码拒绝开发/评价的同路径或同图像名，不检测改名副本或未知预训练重叠。

## 先作开发侧预试，再冻结

### 快速首轮与一条命令运行

先用 [快速配置](configs/drunet_fast.json)：DRUNet、单个高斯核、观测噪声 2.55/255、一个评价种子、完整 BSD68；保留 16/32/64 次预算和四家族公平选参，暂不追加长终端尾部。12 张开发图时，重复计算未消除前共 56,000 次前向；启用精确复用后为 28,160–51,008 次，取决于冻结处方的重复程度。原完整配置每网络约 1,515,264 次。按 4090 上开发图实测约 28 ms/次估计，首轮约 30–60 分钟，具体取决于图像大小及复用数量。这一首轮只覆盖所列单一退化条件。

代码将**完全相同的尺度序列**合并计算：开发侧复用同一候选的完整误差，评价侧在同一观测上复用相同的完整轨迹与指标。候选集合、选参规则、RED 向量场及每种方法的逻辑预算保持原定义，不对近似相同的序列作合并。`reused_from` 标记来源，`executed_network_forwards` 标记该轨迹截至本读数累计执行量（复用行取 0），`network_forwards` 仍描述这条恢复轨迹的预算；不同读数不可重复相加，`completion.json` 汇总真正执行的调用。复用轨迹的计时沿用来源轨迹，不解释为算法本身更快。

在远程仓库根目录执行。非交互 SSH 的系统 `python3` 没装 PyTorch，应明确使用 `/root/miniconda3/bin/python`。已有开发目录 `experiments/assets/data/dev_div2k_12` 是原服务器 DIV2K_train_HR 的前 12 张，预先按名字选定、转灰并等比缩到 481×481 框内，来源记录保存在该目录；它们可能参与过预训练。测试集保持原始整图。

先只估算调用量，不加载权重或图像：

```bash
/root/miniconda3/bin/python -B experiments/code/real_network/run.py estimate --config experiments/configs/drunet_fast.json --development-count 12 --evaluation-count 68
```

一条命令依次检查本地文件、冻结计划、开发选参、评价和配对汇总，使用新的输出目录：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /root/miniconda3/bin/python -u -B experiments/code/real_network/run.py workflow --config experiments/configs/drunet_fast.json --development-images experiments/assets/data/dev_div2k_12 --evaluation-images experiments/assets/dncnn/testsets/BSD68 --dataset bsd68 --out experiments/results/drunet_fast_run --device cuda
```

也可以自行放后台；日志文件和输出目录均换成没有使用过的名字：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nohup /root/miniconda3/bin/python -u -B experiments/code/real_network/run.py workflow --config experiments/configs/drunet_fast.json --development-images experiments/assets/data/dev_div2k_12 --evaluation-images experiments/assets/dncnn/testsets/BSD68 --dataset bsd68 --out experiments/results/drunet_fast_run --device cuda > experiments/results/drunet_fast_run.log 2>&1 &
tail -f experiments/results/drunet_fast_run.log
```

结果在 `drunet_fast_run/evaluation/summary.json`，实际调用在 `evaluation/completion.json`。`workflow` 在开始校准前检查全部图片可读，避免坏图在长校准后才暴露。原来两个已中止目录 `real_network_20261116`、`real_network_fast_20261116` 保留，不作为本次命令的输出目录。

### 用已存开发分数冻结非零尺度历史

首轮三个可调家族均选中终端候选时，使用 `history` 单列机制对照：终端参考保留；其余家族在原来已评估的候选中，按相同开发 MSE、容差和平局规则，选择**实际包含非终端调用的最佳前缀**。只读旧校准的 `plan.json`、`choices.json`、`calibration/scores.csv`，不读取测试误差来选参，也不重新运行开发网格。没有有效非零候选时直接报错，不回退到零前缀。

原 `drunet_fast_run` 中允许零前缀的调优结果继续作为性能比较。新对照回答不同历史的输出/恢复差异，不声称在允许终端候选的家族内误差最优。BSD68 已经被查看，这批是相同观测上的后续配对诊断，不称新盲测。

本轮按原开发分数冻结的处方如下，尺度均为标准差×255；各前缀之后保持终端 5：

| 原预算 | 两端点 | 平台 | 等比退火 |
| ---: | --- | --- | --- |
| 16 | 前缀 8，低 5 做 4 次后高 10 做 4 次 | 10 做 4 次 | 10→5 共 4 次 |
| 32 | 前缀 16，高 25/低 5 按 4 步分块，高端开始 | 25 做 8 次 | 10→5 共 8 次 |
| 64 | 前缀 32，低 5/高 10 每步交替，低端开始 | 10 做 32 次 | 10→5 共 16 次 |

三个预算各有四条不同的完整尺度序列（含终端参考）。默认保留 16/32/64 次读数，**只在原 64 次处方之后追加 32、64 次共同终端更新**，即再读总调用 96、128 次；不在 96/128 重新选择前缀，不能把延长读数称为新预算下调优后的最佳性能。

完整 BSD68、同一原评价噪声种子需要 **47,872 次新前向、0 次新校准**。按已测 28 ms/次，纯推理约 22.3 分钟，含读写和统计约半小时。所有公共求解器参数、权重、核、图像顺序和噪声种子继承原计划，不扩大网络/核/噪声矩阵。

仅检查冻结处方与调用量、无需本机权重或图像，可先执行：

```bash
/root/miniconda3/bin/python -B experiments/code/real_network/run.py history --source-calibration experiments/results/drunet_fast_run/calibration --out experiments/results/drunet_history_preview --plan-only
```

用户同步代码后，用一个**不同的新目录**完成冻结、评价和汇总；预览目录不必同步：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nohup /root/miniconda3/bin/python -u -B experiments/code/real_network/run.py history --source-calibration experiments/results/drunet_fast_run/calibration --images experiments/assets/dncnn/testsets/BSD68 --out experiments/results/drunet_history_run --extra-calls 32 64 > experiments/results/drunet_history_run.log 2>&1 &
tail -f experiments/results/drunet_history_run.log
```

输出中 `selected_prefixes.csv` 给出每个预算/家族的非零处方及相对原无限制选择的开发代价；`calibration/plan.json` 与 `choices.json` 保存派生的冻结规则并保留原选择。`evaluation/records.csv`、`distances.csv`、`summary.json` 仍使用原格式，按 `budget=64, extra_calls=0/32/64` 检查共同终端阶段的输出 RMS、场范数与恢复误差。原结果目录不被修改；源校准也不需要重新运行。

### 完整配置与分阶段运行

默认配置是 [DRUNet](configs/drunet_deblur.json) / [FFDNet](configs/ffdnet_deblur.json)，**是未经真实网络调优的起始配置**：步长 .25、λ=.1、终端 σ=5/255；预算 16/32/64；观测噪声 2.55/255、7.65/255；共同终端尾部至少为预算的 1/4。尾部读数从 K=64 再追加 64、256 次。所有标为 `_255` 的参数均采用 8 位强度单位，内部除以 255；它们都是标准差，不是方差。

先在另存的小开发配置和少量开发图像上确认计时、显存和稳定性，再一次性决定公共求解器设置、预算、候选及独立开发/评价划分。不要把默认配置直接称为公平调优后的最终实验，也不要因测试某方法失败再扩大它的搜索。修改参数后新建输出目录，保留每次计划与选择来源。

固定更新为

\[
x_{k+1}=x_k-\gamma\{A^\top(Ax_k-y)+\lambda[x_k-D_{\sigma_k}(x_k)]\}.
\]

数据项不按观测噪声方差归一化，残差不除以 σ²。周期卷积与其真正伴随用于观测和梯度，不缩放/裁剪图像，不裁剪带噪观测。主比较共用反投影初值、步长、λ 和终端尺度；另一个开发配置可用 `initialization="ridge"`、公共 `ridge` 系数，须分开报告。原生 DPIR/HQS 和各方法独立调优求解器的质量参照不在此固定 RED 主表中。

四个调度家族为终端单尺度、两端点历史、共享平台、标准差等比退火。两端点候选包含高→低、低→高→终端，以及两种相位和块长的交替；只比较保存的候选，不称任意两端点控制。其余三家族都包括终端候选，使用相同候选评估上限；默认每家族每预算 6 个候选，预先固定抽取顺序，终端单尺度 1 个。完整候选序列在计划阶段保存。

每个核/观测噪声/预算分别在开发图像和固定校准噪声上选全局处方；所有候选共用观测。目标为逐图、逐噪声等权的像素 MSE；误差在预设容差 1e−8 内时，依次取较短前缀、较小标准差、模式和块长。这个容差是起始开发设置，不是从评价结果反推的阈值。调用总数保存在 `choices.json`；不提供逐张测试图挑参数的入口。

下面三步分别冻结计划、开发选参、评价；下载完成且开发处方已确定后再执行，均写到新目录：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/run.py plan --config experiments/configs/drunet_deblur.json --images experiments/assets/data/dev --out experiments/results/drunet_calibration
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/run.py calibrate --out experiments/results/drunet_calibration --device cuda
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/run.py evaluate --calibration experiments/results/drunet_calibration --images experiments/assets/dncnn/testsets/BSD68 --dataset bsd68 --out experiments/results/drunet_bsd68 --device cuda
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/summarize.py --out experiments/results/drunet_bsd68
```

Set12 使用同一份冻结选择，独立输出：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/run.py evaluate --calibration experiments/results/drunet_calibration --images experiments/assets/dncnn/testsets/Set12 --dataset set12 --out experiments/results/drunet_set12 --device cuda
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/summarize.py --out experiments/results/drunet_set12
```

FFDNet 把配置改为 `experiments/configs/ffdnet_deblur.json`，输出目录分别改成 `ffdnet_calibration`、`ffdnet_bsd68` 等。每个网络独立做开发选择，不能直接将 DRUNet 选择的尺度称为 FFDNet 已调优。若 CPU 预试，显式用 `--device cpu`；程序不会在 CUDA 缺失时静默回退。

## 输出与解释

- 校准目录：`plan.json` 保存开发图像、公共设置、实际核和全部候选；`calibration/scores.csv` 保留所有候选均值、失败数及调用；`choices.json` 是评价前的选择。文件只创建，不覆盖。
- 评价目录：`plan.json` 保存冻结选择及评价图片；`records.csv` 保存每图、核、噪声种子、预算、家族与追加终端调用的结果；`distances.csv` 保存四家族六对输出 RMS；`completion.json` 标志完整结束。每条记录含 SSE、MSE、PSNR/SSIM、裁剪后质量、弱频带误差、相对终端输出 RMS、场范数、耗时、调用及 CUDA 峰值显存。可加 `--save-states` 保存浮点端点，默认不批量保存整图数组。
- 主误差用未裁剪浮点端点；PSNR 的 data_range=1，逐图计算再平均。SSIM 为 11×11 高斯窗、σ=1.5、总体矩、有效中心均值；PSNR 不裁边。裁剪后质量另列，裁剪不反馈到算法。弱频带由归一化 A 的平方频响 ≤.05 预先定义；只作经验诊断。
- 场范数从最后一次已执行的 RED 更新计算，对应 **更新前状态**，无额外去噪调用；不称输出端点的精确残差。恢复耗时包含完整前缀和共同终端延长，排除读图及离线指标计算；不包含权重加载。没有额外终端最优点搜索。
- `summary.json`、`paired_images.csv` 和 `paired_comparisons.csv` 将重复噪声先在图内汇总，再报告相对终端及全部六对家族的配对差、胜/平/负、中位数和图像 bootstrap 区间（2000 次）。非有限轨迹保留为失败；有失败的组不生成掩盖失败的全图均值。预训练网络的资源错误会中止运行，不能当作一个可忽略图像。

共同终端追加使用锚点预算已经选择的前缀，**不在 K+L 重新调参**。追加后误差差消失，支持该设置下的有限预算效应；持续输出分离加小场范数仍只是所测时长与容差内的经验结果。DRUNet 的 L1 训练不保证 MMSE，FFDNet 的平方损失也不保证精确 MMSE；网络结果不继承 GMM 势、模板竞争、吸引域或风险证书。

## 原因定位的小规模开发测量

首轮负结果及共同终端均值趋近尚未定位原因。[原因核查](results/drunet_history_run/cause_review/analysis.md)分开记录主定理复核、实际适用条件、实现检查和候选覆盖问题。不要将负结果直接归为机制边界，也不要把 λ 改成 1 就称满足定理。

`probe_cause.py` 仅使用已冻结非零历史计划中的前两张开发图，沿用原 64 次处方再延长到 128 次；不读 BSD68，不重新选参。控制噪声使用同一个标准正态样本，只改幅度为 2.55/255 和 7.65/255，全部历史仍沿用原低噪声选择。默认 2166 次网络前向，按已有 28 ms/次约 61 秒纯推理，另加诊断统计时间。用户同步代码后在远程执行，输出目录换成未使用过的名字：

```bash
cd /root/autodl-tmp/annealing
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/probe_cause.py \
  --calibration experiments/results/drunet_history_run/calibration \
  --out experiments/results/drunet_cause_probe \
  --image-count 2 --noise-255 2.55 7.65
```

`forces.csv` 比较同一状态的尺度残差、相对数据项的强度、方向倍增拟合和单步真值误差反事实；真值只作分析，不用于选择规则。`terminal_transport.csv` 把公共尾部的成对距离变化分成数据项、去噪器割线项、步长平方项和浮点余差。`endpoints.csv` 包含相同观测的纯数据更新参考；另报内区/边缘分解，边缘宽度 `min(32, 最短边//4)`，用于检查周期退化与网络填充影响，主误差仍用整图。`denoising_sanity.csv` 检查真实权重对普通 AWGN 的去噪行为。诊断积分器会与原 `red_trajectory` 的 16 步结果逐位核对，调用计入 `completion.json`。

该探查只能定位所用开发图上的作用方向，不是新盲测、完整调参或整个区域的收缩/吸引域证书。代码只读取已有本地资源，不下载文件。仍按用户安排由其同步和运行。

已存结果的代数分解和主定理解析最坏界可在本地重算，无需预训练权重；同样使用新目录：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/analyze_cause.py --source experiments/results/drunet_history_run --preflight experiments/results/real_network_20261116/preflight/gpu_preflight.json --out experiments/results/drunet_history_run/cause_recheck
```

## 必要的小规模实现检查

只用本地生成的合成图像、线性假去噪器和随机初始化网络做 CPU 检查，不需要资源下载，不启动真实数据评价：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/check_implementation.py --out experiments/results/real_network_check_reproduction
```

检查卷积朝向与伴随、固定 RED 公式、调用计数、候选预算/终端尾部、灰度网络形状与本地加载、开发/评价隔离和完整小流程。复跑请换新目录。随机初始化的结构检查不替代下载后的预训练权重兼容和实际恢复验证。
