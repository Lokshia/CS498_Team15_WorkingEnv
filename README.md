# K1 前臂托箱训练环境 · v0.4.0 + stance-v2

## 2026-10-30：真正空手的独立稳站模型（本地分支）

统一操作入口：运行 `.\.venv\Scripts\python.exe -m lifting_lab.mode_viewer --port 8082`，在同一个 viewer 选择空手/托箱、输入重量、应用并重置，再点 Play。[中文操作方法](docs/LOAD_CONTROLS_ZH.md)。切换会重建场景并换对应策略，不模拟连续接箱/卸箱。

`codex/empty-hand-stand` 增加没有箱子实体的空手任务，Actor 只用 72 维本体观测。重新训练后，普通独立评测 **32/32**，小幅速度扰动 **16/16**，CPU 回放 **1/1**；均为仿真、一个训练 seed。模型、配置和原始评测见 [空手模型目录](shared_model/empty-hand-v1-20261030/README.md)，命令见 [中文操作说明](docs/EMPTY_HAND_ZH.md)。空手 viewer 使用 8081，原托箱 viewer 可保留在 8080。

从双臂已抬起的姿态开始，尚未实现抬臂过渡、放箱/卸箱或策略切换，也没有真机 adapter。这是独立任务，不是将托箱模型重量滑块改为 0。本分支尚未推送；下方旧 clone 命令仍获取已发布版本。

## 2026-10-29：平脚与直立姿态版本（本地分支）

本地 `codex/flat-foot-hold` 新增脚底倾角、过度偏向单脚承重、腿姿、躯干直立和软性左右腿镜像奖励，保留原来的托箱与平顺动作要求。当前姿态演示模型在 [stance-v2 模型目录](shared_model/stance-v2-20261029/README.md)，训练与失败记录见 [直立姿态实验说明](docs/STANCE_V2_ZH.md)。同时保留 [stance-v1 平脚版本](shared_model/stance-v1-20261029/README.md) 与[第一阶段诊断](docs/STANCE_HOLD_ZH.md)作为对照。这次改动尚未推送 GitHub；下面原有的 clone 命令仍获取已发布的 smooth-v1 版本。

安装后的本地项目可直接打开新模型：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --support-mode forearms --checkpoint shared_model/stance-v2-20261029/policy.pt --mass 0.4949224923474894 --seed 2027 --seconds 3600 --viewer --paused
```

在 viewer 点击 **Play**。修改 **Payload / 负重** 后点 **应用并重置**，再播放。`policy.pt` 和相邻的 `experiment.json` 要一起保留。v2 两组评测共 94/96 次托箱及严格平脚成功（约 0.25 kg：30/32；约 0.5 kg 和 1 kg：各 32/32）。脚平放、躯干和腿姿有改善，但轻载启动仍出现失败，仿真关节力矩也有所上升；不是所有指标都改善。v1 的已测启动可靠性更好。两版均不代表完全对称站姿或真机部署已验证；本次没有修改物理参数或降低托箱成功标准。

## 队友快速开始：运行已训练的平稳托箱模型

本分支 `codex/smooth-forearm-hold` 包含代码、精选 checkpoint 和评测结果。先安装 [Git](https://git-scm.com/downloads) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)（uv ≥ 0.11.21），再在 Windows PowerShell 运行：

```powershell
git clone --branch codex/smooth-forearm-hold --single-branch https://github.com/Lokshia/CS498_Team15_WorkingEnv.git
cd CS498_Team15_WorkingEnv
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Backend cu130
```

只看演示、没有可用 NVIDIA GPU 时，把安装参数 `cu130` 改成 `cpu`；下面的 CUDA 评测和续训需要 GPU 安装。Linux 安装见第 1 节。

**打开已训练模型的 CPU viewer**（安装 CUDA 版也可以用 CPU 预览）：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --support-mode forearms --checkpoint shared_model/smooth-v1-20261028/policy.pt --mass 0.4949224923474894 --seconds 3600 --viewer --paused
```

打开终端打印的网址，通常为 `http://127.0.0.1:8080/`，点击 **Play**。在 **Payload / 负重** 中修改重量，再点 **应用并重置**。保留终端运行。必须带上上面的 `--checkpoint`；省略它运行的是零动作 PD，不是已训练策略。

**CUDA 正式五负载评测**（每档 50 个 20 秒 episode；失败可提前终止）：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab evaluate --checkpoint shared_model/smooth-v1-20261028/policy.pt --protocol configs/smooth_hold_validation.json --device cuda --num-envs 50 --output outputs/smooth_shared_five_load_eval
```

该模型训练 seed 为 11，协议自动选评测 seed 5011。已有结果为四档约 0.25–1 kg 离散负载各 **50/50**，1.48 kg 压力测试 **33/50（66%）**；这是一个训练种子的结果，没有测试推力或摩擦随机化。启动阶段仍有瞬态动作，不能称为全程无抖动。详见 [模型与结果文件](shared_model/smooth-v1-20261028/README.md) 和 [训练说明](docs/SMOOTH_HOLD_ZH.md)。

**从发布模型续训 100 次 PPO 更新**（可选，会实际使用 GPU）：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --support-mode forearms --resume shared_model/smooth-v1-20261028/policy.pt --num-envs 512 --iterations 100 --seed 11 --output outputs/smooth_shared_seed11_more
```

续训继承原模型的质量、`smooth-v1` 奖励、rollout 长度和优化器状态；它不是新的独立训练种子，也不保证续训后性能提高。评测和训练的输出目录必须使用新名字。`policy.pt` 与同目录 `experiment.json` 要一起保留。

**分享方式：** 此 Git 分支含 `shared_model/smooth-v1-20261028/` 的精选模型和结果；`scripts/package_release.py` 生成的 ZIP 仍是纯源码包，不含模型或视频。只拿到源码 ZIP 的队友需要另取这个完整模型目录。

四位队友共用的 mjlab / MuJoCo-Warp 环境。默认演示把自由方块放在近水平的两条前臂上，靠接触和摩擦承重，可以在浏览器改重量，也支持 CUDA。**方块没有绑在手上；它可以滑动、倾斜、掉落，机器人也可能倒下。不学习从地面拾取。** 不传 checkpoint 的演示仍是零动作 PD，不是训练好的策略。

2026-10-28 增加了可选 `--reward-profile smooth-v1`、关节诊断和复测协议，用来减少持续抽动。PPO、97 维观测、22 个动作、原始位置控制、物理参数和接触成功判据均保留。训练方法、已测范围及限制见 [平稳托箱说明](docs/SMOOTH_HOLD_ZH.md)；新训练仍须显式选择该奖励，默认 `baseline` 保留对照行为。模型回放/续训会继承相邻 `experiment.json` 的奖励配置。

本项目基于 v0.4.0：前臂自由接触、PPO 训练、续训、负载课程和独立评测。团队请使用同一个日期版本的代码包或同一 Git commit；旧 v0.4.0 ZIP 没有新奖励配置。源码包不包含权重，分享训练结果时须同时携带 `policy.pt` 与相邻 `experiment.json`。

原始 v0.4.0 的历史验收：Windows / RTX 5090 Laptop GPU，32 并行环境、24 步 rollout，共 10 次 PPO 更新（含续训、三阶段课程和盲质量对照），总计 7,680 个训练 transitions；保存/恢复、有限值、优化器延续、评测和 CPU 推理检查通过。当时回归 160 项通过、1 项因 Windows 符号链接权限跳过。**当时短跑 conditioned / blind / 零动作基线的 quick 评测均为 0/6 成功，只是流程验证。** 后续平稳托箱训练的结果独立记录在上述说明和对应结果包中，不能与这次历史验收混淆。

## 1. 安装（只做一次）

Git 用户进入 clone 得到的 `CS498_Team15_WorkingEnv` 目录；源码 ZIP 用户先**完整解压**，进入里面的 `lifting-lab` 文件夹。下面都在项目根目录打开终端运行，不要在压缩包内运行。需要联网，并先装好 [Git](https://git-scm.com/downloads) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)（uv ≥ 0.11.21）；安装后重新打开终端。

### Windows / PowerShell

有 NVIDIA 显卡：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Backend cu130
```

没有可用 NVIDIA 显卡、只看演示：把上面的 `cu130` 改为 `cpu`。

### Linux x86-64

```bash
bash scripts/bootstrap.sh cu130
```

无 NVIDIA 显卡时同样改成 `cpu`。本分支的已记录训练和 GPU 评测来自 Windows / RTX 5090 Laptop，不能代替其他硬件上的复测或真机验证。**Linux 安装和显示尚未在 Linux 实机验证**。此发行版不支持 macOS / ARM。

脚本会安装固定的 Python 3.13.14 和 `uv.lock` 中的依赖，创建当前项目自己的 `.venv`，最后检查 CPU/GPU。无需安装旧 HW3，也不用手动激活环境。CUDA 13.0 配置需要兼容的 NVIDIA 驱动（R580 及以上；`nvidia-smi` 右上角 CUDA Version 低于 13.0 时 `cu130` 的 doctor 会直接失败，先升驱动或改用 `cpu`）；检查失败不会偷偷切换 CPU。首次下载会比较大，ZIP 本身不包含依赖和机器人资产。`uv.lock` 只解析 Windows x86-64 和 Linux x86-64；GH200 这类 aarch64 节点需要另行解锁依赖，不要在那里直接跑本脚本。

## 2. 打开机器人（每次使用）

Windows：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --support-mode forearms --checkpoint shared_model/smooth-v1-20261028/policy.pt --mass 0.4949224923474894 --seconds 3600 --viewer --paused
```

Linux：

```bash
./.venv/bin/python -m lifting_lab demo --device cpu --support-mode forearms --checkpoint shared_model/smooth-v1-20261028/policy.pt --mass 0.4949224923474894 --seconds 3600 --viewer --paused
```

打开终端打印的地址，通常是 `http://127.0.0.1:8080/`。保留终端运行。单机器人预览使用 CPU 即可；安装了 CUDA 配置也可以运行这个命令。要测试 GPU 路径，可将 `--device cpu` 改成 `--device cuda`。

- **Payload / 负重 → 重量 (kg) → 应用并重置**：按当前模式重建质量和惯量，回到初始支撑姿态并暂停。
- **Play / Pause**：运行或暂停；**Step**：单步；**Reset Environment**：重置。
- 前臂交互预览允许箱子掉落、机器人倒地后继续模拟，不自动重置，也不计作评测结果；看完后手动 Reset。普通 smoke / 离线 demo 保留跌倒终止条件。
- 窗口中的 0–10 kg 只是仿真输入范围，**不代表真机承载额定值**。0 kg 实际用 0.001 kg，避免零惯量。
- 上面的命令加载发布的平稳托箱模型。只有主动省略 `--checkpoint` 才是零动作 PD 基线；它不是平衡控制器，箱子可能滑落、机器人可能倒下。不要把暂停时的摆放当成稳定承重结果。
- 录像前另装 FFmpeg 并加入 PATH（`ffmpeg -version` 验证）；不录像则不需要。录像期间须先停止录像再改重量。

`demo`、`smoke`、`train`、`evaluate` 的 `--support-mode` 都可选 `forearms`（默认，前臂自由接触）或 `pregrasp`（旧版双点连接）。切换模式需重新运行命令；重量控件只改变当前模式的负载。

模式与 checkpoint 不可混用。前臂 checkpoint 可以在 viewer 播放；旧连接策略必须明确选 `--support-mode pregrasp`。同为 97 维观测不代表任务相同。只加载自己或可信队友生成的权重，并保留相邻的 `experiment.json`。

## 3. 小测试 → 自己训练 → 评测

以下使用 Windows；Linux 将解释器换为 `./.venv/bin/python`。所有输出目录必须是新名字，程序不覆盖旧实验。安装和打开 viewer 不会自动启动训练。

**先跑端到端小测试**（会实际训练少量迭代，不是性能评测）：

```powershell
.\.venv\Scripts\python.exe scripts/check_training.py --device cuda --num-envs 32
```

脚本依次检查 PPO、保存/恢复优化器、三阶段课程、mass-blind、快速评测和 CPU checkpoint 推理。日志和 `acceptance.json` 放在终端打印的 `outputs/..._training_acceptance/`。PASS 只说明流程与数值检查通过，不表示学会托箱。

**自行开始研究训练**：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --support-mode forearms --num-envs 512 --seed 11 --iterations 100 --curriculum --output outputs/forearms_seed11
```

这是预算示例，不是已验证收敛配方。`--curriculum` 依次使用机器人质量的 0%、2.5%、5%，每阶段 100 次 PPO 更新，延续模型和优化器。0% 使用 1 g 数值代理。每次 rollout 默认 24 步，训练 transitions = 环境数 × rollout 步数 × 迭代数 × 阶段数。显存不足可减少环境数，但团队对比须对齐采样预算。

单一重量用 `--mass 0.5` 或 `--mass-fraction 0.025`，不要同时带 `--curriculum`。做盲质量对照加 `--mass-blind`，保持其余预算相同，另取输出名。用 11、22、33 三个独立 seed，不能通过续训换 seed 冒充独立实验。

**续训、快速评测、回放**：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 512 --seed 11 --iterations 100 --mass-fraction 0.05 --resume outputs/forearms_seed11/phase_2/policy.pt --output outputs/forearms_seed11_more
.\.venv\Scripts\python.exe -m lifting_lab evaluate --device cuda --num-envs 32 --quick --checkpoint outputs/forearms_seed11_more/policy.pt --output outputs/eval_seed11_quick
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --mass-fraction 0.05 --checkpoint outputs/forearms_seed11_more/policy.pt --viewer --paused --seconds 3600
```

盲质量 checkpoint 的续训、评测和回放都加 `--mass-blind`。`--quick` 仅 6 个两秒 episode，不能算 proposal 正式结果。正式评测去掉 `--quick` 并另换输出名：每个 checkpoint 在 3 种负载下各测 50 个 20 秒 episode。

默认协议为 `configs/team_protocol.json`，任务版本 `forearms-contact-v1`：不仅检查姿态和高度，还检查双侧前臂真实接触、连续失去承托、箱体与地面/其他机器人部位接触。前臂末端球可辅助接触，但不算有效前臂承托。具体阈值和边界见 [团队协议](docs/TEAM_PROTOCOL.md)。奖励是可调整的起点，不保证收敛；修改判据或物理后须更新任务版本并重新验证。

**可选：训练时随机化负载和加扰动**（默认全部关闭，不加这些参数就是上面的固定质量任务）：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 512 --seed 11 --iterations 300 --mass-range-fraction 0 0.05 --com-offset-m 0.02 0.03 0 --payload-xy-jitter-m 0.01 --payload-yaw-jitter-rad 0.02 --push-interval-s 3 5 --push-velocity-m-s 0.3 --output outputs/forearms_dr_seed11
```

- `--mass-range-fraction LOW HIGH`：每个环境每次 reset 在该区间（机器人质量比例）均匀抽一个箱子质量，直接写进该环境的 Warp 模型（质量 + 均匀箱惯量一起改），不重建环境。与 `--mass` / `--mass-fraction` / `--curriculum` 互斥；元数据里的 nominal 质量是区间中点。观测第 94 项仍是该环境的真实质量比，`--mass-blind` 照旧置零。
- `--com-offset-m X Y Z`：重心在箱体坐标系内均匀偏移的半幅（米），相当于配重没放正；碰撞箱不变，偏移必须小于半边长。
- `--payload-xy-jitter-m` / `--payload-yaw-jitter-rad`：reset 时只移动箱子相对前臂的位置：x 只向前（箱子后沿离躯干只有 1 cm），y 和 yaw 对称。已验证上限 0.01 m / 0.03 rad，超过后箱子会落在末端球上、起步就失去双前臂承托。
- `--push-interval-s LOW HIGH` + `--push-velocity-m-s V`：每个环境按自己的计时器每隔 LOW–HIGH 秒给机器人基座水平速度各加一个 ±V 的随机增量（Isaac Lab 同款速度踢），箱子只通过接触受影响。两个参数必须同时给。
- `--resume` 默认沿用 checkpoint 里记录的这些设置，命令行显式给出的参数才会覆盖；`experiment.json` 的 `randomization` 和 `mass.mass_range_fraction` 记录实际使用值。
- 带推力评测用 `configs/team_protocol_push.json`（3–5 s 一次、0.3 m/s）：`evaluate --protocol configs/team_protocol_push.json --checkpoint ...`。它与无推力协议分开报告，不能合并成功率。若粗略按固定水平面上的滑块计算，取间距 d=1 cm、μ=0.7，停止距离估算给出 v=√(2μgd)≈0.37 m/s。这忽略了机器人运动、箱体转动及瞬态接触，不能作为本系统的硬速度上限，也不能据此断言碰撞与平衡能力无关；0.3 m/s 是待验证的协议设定，实际结果须由仿真和后续标定判断。

这些只改变训练分布和评测条件，不改变观测/动作/成功判据，所以 `task_schema` 仍是 `forearms-contact-v1`，checkpoint 可以在有无随机化的环境间互相加载；报告结果时要写明训练用的随机化范围和评测协议。零动作 PD 基线本身在 0.5 kg 下约 0.16 s 后箱子就会后滑碰到躯干，这不是随机化造成的；机器人本体在零动作下也会在 1.5 s 内前倾倒地（踝/髋是软 PD，没有策略就没有平衡），所以任何策略都要先学会站。

**三个基线 run 一次跑完**（站+托 1 g 箱子、顺序质量课程、每局随机质量，各自接协议评测，固定文件夹名，最后打印成功率表和该分享的 checkpoint 路径）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\milestones.ps1 -Device cuda -NumEnvs 512 -Iterations 500
```

`-Iterations` 是每个 run（课程为每阶段）的 PPO 更新数；加 `-Quick` 只验证流程。不含推力和负载随机化。

## 4. 四个人如何保持一致

所有人解压同一个 ZIP，或检出团队仓库的同一 commit，不自行升级依赖或修改 `uv.lock`。不要运行裸 `uv sync`，也不要借用其他项目的 Python。安装脚本已明确选择 `cu130` 或 `cpu`。

Windows 的 GPU 物理检查：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab smoke --device cuda --support-mode forearms --num-envs 32
```

Linux 将解释器路径换成 `./.venv/bin/python`。CPU 用户用 `--device cpu --num-envs 2`。需要对照旧模式时明确改成 `--support-mode pregrasp`。这只检查仿真流程，不代表稳定托箱或训练成功。

普通回归测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

额外真实 CPU / GPU 测试分别由环境变量 `LIFTING_TEST_ENV=1`、`LIFTING_TEST_GPU=1` 开启。不同硬件不保证逐位相同；比较时记录模式、任务版本、质量、seed、环境数、checkpoint、代码与协议 hash。四人复测同一个 checkpoint 时可用 `scripts/compare_results.py` 检查结果文件的配置是否一致；这个工具不用于给不同策略排名。

## 5. 文件安排

```text
lifting-lab/
  README.md                 本文件：安装与日常操作
  pyproject.toml, uv.lock    固定依赖及 CPU / CUDA 配置
  src/lifting_lab/           物理模型、viewer、训练和评测代码
  configs/                  前臂托箱协议及独立的 legacy 协议
  scripts/                  安装、结果比较、打包与校验工具
  tests/                    自动化检查
  docs/                     物理与训练设计、团队评测、实物对应边界
  shared_model/smooth-v1-20261028/  Git 分支附带的精选模型、评测和诊断
  THIRD_PARTY_NOTICES.md     依赖来源与授权边界
  RELEASE_MANIFEST.json     仅源码 ZIP 生成的包内文件 SHA-256 清单
```

运行后自动生成的 `.venv/`、`outputs/` 不随 Git 发布。`shared_model/smooth-v1-20261028/` 包含精选 checkpoint、相邻训练配置、五负载评测、约 0.495 kg 演示视频、关节诊断及 `SHA256SUMS.json`；详见该目录的 [文件说明](shared_model/smooth-v1-20261028/README.md)。`scripts/package_release.py` 的源码 ZIP 不包含这个目录。凭据、旧日志和第三方模型资产均不随本分支发布；依赖和机器人资产在安装时从锁定的上游提交获取，请遵守 [来源说明](THIRD_PARTY_NOTICES.md)。

需要理解“前臂接触和旧手部连接有什么不同”：看 [实物与仿真对应](docs/SIM_TO_REAL_ZH.md)。需要详细参数：看 [环境设计](docs/ENVIRONMENT_DESIGN.md) 或运行 `python -m lifting_lab COMMAND --help`（使用本项目解释器）。本包不确认真实前臂外壳、衬垫或托架可以承重。

## 6. 用 Git 给四个人共用

团队仓库为 [CS498_Team15_WorkingEnv](https://github.com/Lokshia/CS498_Team15_WorkingEnv)，平稳托箱版本在 `codex/smooth-forearm-hold` 分支。请按开头命令 clone，在这个独立仓库内操作。不要在包含其他课程作业的父目录执行 `git add .`。该分支供队友复测和继续研究；合入 `main` 应另行审查。

- 保留 `uv.lock`、`.python-version`、`.gitignore` 和中文 README。
- `.gitignore` 排除 `.venv/`、`outputs/`、`validation/`、日志和本地凭据；仅为指定的 `shared_model/smooth-v1-20261028/` 发布文件保留例外。
- 提交前检查 `git status --short`，确认只有本项目的源码、说明及有意发布的模型/结果；不要使用 `git add -f` 上传整个输出目录。
- 队友 clone 后进入仓库，按本文件第 1 节安装。GPU 用户选 `cu130`，CPU 用户选 `cpu`。
- 开始统一测试前对齐 `git rev-parse HEAD` 的结果，避免各自使用不同版本。
- `RELEASE_MANIFEST.json` 只对应那一次 ZIP 内容。修改代码后应重新打包生成清单，不把旧清单当成当前代码的校验结果。

只分享本项目源码及有意发布的模型/结果；不上传 `.venv` 中下载的第三方实现/机器人资产，详见来源说明。新实验保存在 `outputs/`；以后要发布新的模型时，另建有日期的精选目录，保留配置、评测和校验清单。

## 7. 给 GPT / 队友的说明

> 请先读本 README、pyproject.toml 和 docs/ENVIRONMENT_DESIGN.md。这是独立的 K1 前臂托箱研究项目，不要修改其他课程环境。使用当前项目 .venv 的 Python。所有 CLI 默认 forearms，legacy pregrasp 须显式选择，不混用 checkpoint。先检查安装与小规模训练流程，再根据我授权的预算启动训练；不操作实体机器人，不升级依赖。不要把短测试、viewer 或高奖励当成 20 秒成功率；使用独立评测。物理/判据变化须版本化，实物对应先读 docs/SIM_TO_REAL_ZH.md。

遇到问题时提供完整命令、报错、系统和 GPU 型号。端口被占用时看终端实际打印的地址；不要随意结束别人的 Python 进程。离线、仓库访问失败或驱动不兼容时，保留错误，勿用重新生成 lock 的方式绕过。
