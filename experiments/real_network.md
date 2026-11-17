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

先前的[原因核查](results/drunet_history_run/cause_review/analysis.md)分开记录主定理复核、实际适用条件、实现检查和候选覆盖问题；现有[两图探查及原因分解](results/drunet_cause_probe/analysis.md)定位了当前开发设置的作用方向与消差来源。不要将负结果直接归为机制边界，也不要把 λ 改成 1 就称满足定理。

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

保存的探查无需再次执行网络即可汇总并出图；这个离线分析入口只导入 NumPy/Matplotlib 等分析依赖，不导入 PyTorch。复算写新目录：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/summarize_cause.py --source experiments/results/drunet_cause_probe --out experiments/results/drunet_cause_probe/review_recheck
```

可在原图像仍位于计划路径的机器上，重构同观测纯数据参考的无噪声恢复项、实际噪声响应项及交叉项；无去噪器调用，不重新生成噪声实现。此项显式采用双精度算子，并报告与原单精度 GPU 端点的差值，不替换原实验：

```bash
/root/miniconda3/bin/python -B experiments/code/real_network/data_reference.py --source experiments/results/drunet_cause_probe --out experiments/results/drunet_data_reference_recheck.json
```

已存结果的代数分解和主定理解析最坏界可在本地重算，无需预训练权重；同样使用新目录：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/analyze_cause.py --source experiments/results/drunet_history_run --preflight experiments/results/real_network_20261116/preflight/gpu_preflight.json --out experiments/results/drunet_history_run/cause_recheck
```

## 求解器与尺度时机的定向修正

[离线干预分析](results/drunet_solver_repair/analysis.md)已从保存的作用力标量计算减轻/增强 λ 的方向，并准备 `repair_solver.py` 检验完整轨迹。它先用前两张既有开发图校准公共 γ/λ/终端 σ，再固定公共求解器比较短尺度段早放/晚放，最后在第 3/4 张图和两组新噪声上确认。每条轨迹均 64 次调用，末尾至少 16 次共同终端更新。两端的误差变化分别对原终端和校准终端报告，便于区分求解器修正与尺度历史收益。

默认总计 13,568 次前向，按旧计时约 6.33 分钟纯推理。先只生成计划和调用估算时加 `--plan-only`；该目录只作预览，正式运行另用新目录。用户同步代码后在远端运行：

```bash
cd /root/autodl-tmp/annealing
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/repair_solver.py \
  --source-calibration experiments/results/drunet_fast_run/calibration \
  --out experiments/results/drunet_repair_run
```

`plan.json` 先保存 27 个公共求解器候选、全部尺度历史构造、图像/噪声划分与调用估算。`solver_choices.json`、`history_choices.json` 在确认图加载前写出；`constant_fit.csv`、`history_fit.csv` 保留所有开发候选。`check_summary.csv` 是确认结果，包含相对原终端及校准终端的误差变化，配对时机对照为 `best_delayed` 与 `paired_early`；这两条使用完全相同的尺度用量。非零诊断策略保留，性能选择仍允许终端胜出。非有限轨迹保留为失败，不纳入掩盖失败的均值。

相同状态上的 λ 干预与割线阈值可离线复算，无网络调用，输出到新目录：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/analyze_strength.py --source experiments/results/drunet_cause_probe --out experiments/results/drunet_strength_recheck
```

## 冻结处方在剩余开发图上的验证

`validate_repair.py` 从已有 `performance_selection` 直接读取处方，在原开发顺序剩余 8 图上验证。默认新种子为 2026111713/2026111714；跨观测噪声幅度仍共享标准正态场。四条策略为原终端、校准终端、冻结历史、相同尺度用量换放置时机的对照。统一 64 次，不重新选参或按验证误差更换性能策略。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/validate_repair.py \
  --source experiments/results/drunet_repair_run \
  --development-plan experiments/results/drunet_fast_run/calibration/plan.json \
  --out experiments/results/drunet_frozen_validation
```

默认 8192 次前向，按 28 ms/次约 3.82 分钟纯推理。这些是既有开发资源中未参与本轮修正选择的剩余图，不是预训练未见数据或 BSD68 新盲测。输出 `plan.json` 先冻结所有处方和噪声；`records.csv`、`summary.csv` 保存固定预算结果，`paired_images.csv` 先在图内平均种子，`paired_comparisons.csv` 给配对误差、图像胜负及 2000 次图像重采样区间。重采样单位为图像，区间描述这批固定开发图的变动，不据此保证 BSD68 总体效果。复跑换新目录。

[实际八图结果及采用的配置](results/drunet_frozen_validation/analysis.md)。同一入口可将冻结策略原样应用到标准集合；不会再次开发选参。已准备的 BSD68 处方采用新种子 2026111715，两个噪声档和四条策略，共 34,816 次前向。当前只生成过计划预览，正式运行使用新的输出目录：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/validate_repair.py \
  --source experiments/results/drunet_repair_run \
  --development-plan experiments/results/drunet_fast_run/calibration/plan.json \
  --images experiments/assets/dncnn/testsets/BSD68 --dataset bsd68 \
  --seeds 2026111715 --out experiments/results/drunet_bsd68_frozen_run
```

只预览时增加 `--plan-only` 并另取目录名。原 BSD68 已在早期实验中查看，冻结配置和新噪声评价不称新盲测；源处方不会因本次八图结果改变。

## 修正配置的共同终端延长

`probe_terminal.py` 读取修正实验已冻结的公共求解器和性能历史，仅比较校准终端与冻结历史。沿用原确认图、种子、核、初值和两档同样本噪声；不重新选处方。若原端点未保存，只重放前 64 次，然后共同终端延长到 128/256/512/1024 次。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/probe_terminal.py \
  --source experiments/results/drunet_repair_run \
  --out experiments/results/drunet_terminal_probe --max-calls 1024
```

默认 16,400 次前向。`records.csv` 给出实际输出状态的 MSE/SSE、PSNR、RED 场 RMS/最大分量及数据项、正则项范数；场范数对应输出本身。计算出的场复用于下一次更新，每条轨迹仅在最终端点多一次诊断调用。`pairs.csv` 给出历史与终端的距离、下一步距离比、割线方向系数及数据/正则/步长平方的距离变化项。网络未假定存在显式势，割线系数不作为整个邻域的稳定性证书。`states/` 保存起点和最终端点，可直接续算。

若 1024 次仍有持续距离或方向放大，可从同一状态继续。下例保留两张图，只取原顺序第一个种子，分别追踪两档噪声至 4096 次；属于原观测上的机制诊断，不重新选调度或改写确认性能表：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/probe_terminal.py \
  --source experiments/results/drunet_repair_run \
  --resume-from experiments/results/drunet_terminal_probe \
  --out experiments/results/drunet_terminal_long --max-calls 4096 \
  --image-count 2 --seed-count 1
```

续算两图两档共 24,584 次新前向，不重放前缀。输出目录必须未使用。离线汇总只读取保存表格，不加载网络：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/analyze_terminal.py --source experiments/results/drunet_terminal_probe --out experiments/results/drunet_terminal_probe/review
```

第一张高噪声图在 4096 次出现低残差、持续分离的近驻点候选。[局部返回探查](results/drunet_attraction_probe/analysis.md)从这些端点及其正负四分之一连接方向扰动、中点出发，继续相同终端更新；返回距离与同时推进的原端点比较。它使用 7203 次前向，包含五尺度同状态作用力测量：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/probe_attraction.py \
  --source experiments/results/drunet_terminal_long \
  --out experiments/results/drunet_attraction_probe --steps 1024 --noise-255 7.65
```

这是已查看观测上的机制探查；扰动不使用真值，不按误差或场范数在线选控制，不将离散返回起点当成完整区域证明。真实运行结果见 [共同尾部分析](results/drunet_terminal_probe/analysis.md)、[4096 次续算](results/drunet_terminal_long/analysis.md)。复跑更换新输出目录。

## 一条命令运行本轮定向工作

入口为 [run_directed.py](code/real_network/run_directed.py)。用户同步代码后自行运行；默认先做三尺度实际路径干预，再做 BSD68 冻结评价，两项串行，全部子计划在推理前写出。需要保留已有 `drunet_repair_run`、`drunet_fast_run/calibration`、`drunet_terminal_long`（含 `states/`）、原权重和图像。

```bash
cd /root/autodl-tmp/annealing
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/run_directed.py \
  --out experiments/results/drunet_directed_run
```

默认实际前向合计 **47,073 次**，按 28 ms/次约 **21.97 分钟纯推理**，另加加载、指标和写盘开销。输出目录必须未使用；预览使用另外的目录名并加 `--plan-only`。可加 `--phase path` 或 `--phase quality` 只做一项，不同运行各用新目录。

### 三尺度实际路径干预

[probe_path.py](code/real_network/probe_path.py)使用原顺序第一张确认图、原第一个种子、观测噪声 7.65/255。先执行 32 次 σ10 得到唯一共享状态，然后各执行 16 次 σ10/25/49，再共同 σ10 直到总计 4096 次。γ、λ、观测、反投影初值和更新式均继承已冻结配置。只复用前 32 次相同调用；没有按真值挑分支、切换时机或停止点。

三分支各有 4096 次逻辑恢复预算。共享前缀后恢复实际调用 12,224 次，额外场诊断 33 次，共 **12,257 次前向**，约 **5.72 分钟纯推理**。脉冲内每步保存路径指标，此时真正的终端场是额外诊断，实际更新仍由该分支的干预尺度驱动；脉冲结束后各分支都使用同一终端场。第 32 次的共同终端测量也精确复用。

`path_probe/` 输出：

- `records.csv`：逐路径读数的 MSE/SSE、真实终端场、实际更新方向、相对两个已存端点的距离、投影坐标和正交位移。投影只描述轨迹，不单独作为跨区证据。
- `pairs.csv`：三对分支的距离和共同终端割线分解；脉冲期间的下一步终端距离比明确标为反事实，脉冲结束后才与实际下一步一致。
- `summary.csv`、`final_pairs.csv`：相同最终预算下的三分支与三对比较。`closest_reference` 仅表示更近的已存端点；应同时查看距离比例与场残差，不能把任意远处状态强行归到某个吸引域。
- `states/`：共同第 32 次状态，各分支第 48/64/4096 次状态，可供后续分析。已有 4096 次端点是距离参考，不用于选择控制或替代新分支的推进。

单独运行路径干预也可使用：

```bash
/root/miniconda3/bin/python -u -B experiments/code/real_network/probe_path.py \
  --source experiments/results/drunet_terminal_long \
  --out experiments/results/drunet_path_probe --max-calls 4096
```

### BSD68 固定质量评价

`bsd68/` 使用完整 68 图、两档观测噪声、新种子 2026111715，保留原终端、校准终端、冻结历史、同量换时机四条策略，各 64 次，共 **34,816 次前向**，约 **16.25 分钟纯推理**。不使用路径干预结果重新选择尺度历史。评价记录按图实时写出，非有限记录保留；正常结束后生成 `summary.csv`、`paired_images.csv`、`paired_comparisons.csv`。根目录 `completion.json` 表示两项均完成，各子目录也有自己的完成记录。

## 三个新种子的尺度强对照

[validate_scales.py](code/real_network/validate_scales.py)读取已完成的 `drunet_directed_run/bsd68/plan.json`，只读取冻结处方与图像顺序，不读取其评价分数。固定完整 BSD68、两档噪声、三个新种子 **2026111716/2026111717/2026111718**，比较三个策略：

- `tuned_terminal`：沿用校准终端，γ=.75、λ=.1，低/高噪声终端 σ2.5/10。
- `history_25`：沿用原先已选历史，低噪声前 16 次 σ25 后回终端，高噪声为 32 次 σ10、16 次 σ25 后回终端。
- `history_49`：只将上述脉冲尺度从 25 换成 49，位置、次数、公共求解器和终端完全相同。

每条轨迹仍为 64 次。同一图/种子跨方法和观测噪声幅度共享标准正态场，保留原图像索引。更新、数据项与残差口径不变，不重新调参、按图选择或按误差提前停止。此轮不增加新核、第二网络或长终端尾部。

用户同步 `experiments/code/real_network/` 后，在远端仓库根目录运行；沿用已有 BSD68、权重及 `drunet_directed_run/bsd68`，不需要重新准备数据：

```bash
cd /root/autodl-tmp/annealing
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/root/miniconda3/bin/python -u -B experiments/code/real_network/validate_scales.py \
  --out experiments/results/drunet_scale_repeat
```

默认 **78,336 次前向、1224 条评价记录**，按 28 ms/次约 **36.56 分钟纯推理**，另加加载、指标和写盘开销。输出目录必须未使用。仅查看计划时加 `--plan-only`，并另用 `drunet_scale_repeat_preview` 等新目录；预览不会加载网络或图像。

输出说明：

- `plan.json`：源计划、全部三条处方、图像顺序、新种子及调用量，在推理前冻结。
- `records.csv`：逐图、种子、噪声、策略的原始结果，按图落盘。
- `seed_summary.csv`、`seed_comparisons.csv`：每个新种子的均值、三个配对比较、逐图胜负与配对图像重采样区间。
- `summary.csv`、`paired_images.csv`、`paired_comparisons.csv`：先在每图内平均三个种子，再等权汇总 68 图。比较为 σ25 对终端、σ49 对终端、σ25 对 σ49；同时给出每种子改善幅度的最小/最大值。
- `completion.json`：实际调用量、记录数、失败轨迹数及运行计时。

合并区间使用 2000 次配对图像重采样，保留每张图的三个噪声结果，不把 204 个图像/种子组合当成 204 张独立图像。各种子结果另外报告。若出现非有限轨迹，原始记录保留，完整均值汇总不掩盖失败。BSD68 已在此前查看，本轮是固定处方的新噪声复验，不声明新盲测或预训练未见数据。

## 必要的小规模实现检查

只用本地生成的合成图像、线性假去噪器和随机初始化网络做 CPU 检查，不需要资源下载，不启动真实数据评价：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/real_network/check_implementation.py --out experiments/results/real_network_check_reproduction
```

检查卷积朝向与伴随、固定 RED 公式、调用计数、候选预算/终端尾部、灰度网络形状与本地加载、开发/评价隔离和完整小流程。复跑请换新目录。随机初始化的结构检查不替代下载后的预训练权重兼容和实际恢复验证。
