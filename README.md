# K1 前臂托箱训练环境 · v0.4.0

四位队友共用的 mjlab / MuJoCo-Warp 环境。新默认演示把自由方块放在近水平的两条前臂上，靠仿真碰撞体之间的接触和摩擦承重，可以在浏览器改重量，也支持 CUDA。**方块没有绑在手上；它可以滑动、倾斜、掉落，机器人也可能倒下。没有训练好的平衡策略，也不学习从地面拾取。**

本说明对应 v0.4.0：前臂自由接触、PPO 训练、续训、负载课程和独立评测。团队请统一使用 `k1-lifting-lab-v0.4.0.zip` 或同一 Git commit。旧包不会自动更新；本包不附带训练成功的策略。

本版在 Windows / RTX 5090 Laptop GPU 上完成小规模验收：32 并行环境、24 步 rollout，共 10 次 PPO 更新（含续训、三阶段课程和盲质量对照），总计 7,680 个训练 transitions；保存/恢复、有限值、优化器延续、评测和 CPU 推理检查通过。完整回归 160 项通过、1 项因 Windows 符号链接权限跳过。**短跑 conditioned / blind / 零动作基线各自的 quick 评测均为 0/6 成功；这只是流程验证，未达到稳定托箱目标。** 未进行正式 20 秒评测、收敛验证或真机测试。

## 1. 安装（只做一次）

先把 ZIP **完整解压**，进入里面的 `lifting-lab` 文件夹，再在这里打开终端。不要在压缩包内运行。需要联网，并先装好 [Git](https://git-scm.com/downloads) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)（uv ≥ 0.11.21）；安装后重新打开终端。

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

无 NVIDIA 显卡时同样改成 `cpu`。已有运行环境在 Windows RTX 5090 上使用；这不等于新前臂模式已经通过平衡或真机验证。**Linux 安装和显示尚未在 Linux 实机验证**。此发行版不支持 macOS / ARM。

脚本会安装固定的 Python 3.13.14 和 `uv.lock` 中的依赖，创建当前项目自己的 `.venv`，最后检查 CPU/GPU。无需安装旧 HW3，也不用手动激活环境。CUDA 13.0 配置需要兼容的 NVIDIA 驱动；检查失败不会偷偷切换 CPU。首次下载会比较大，ZIP 本身不包含依赖和机器人资产。

## 2. 打开机器人（每次使用）

Windows：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --support-mode forearms --mass 0.5 --seconds 3600 --viewer --paused
```

Linux：

```bash
./.venv/bin/python -m lifting_lab demo --device cpu --support-mode forearms --mass 0.5 --seconds 3600 --viewer --paused
```

打开终端打印的地址，通常是 `http://127.0.0.1:8080/`。保留终端运行。单机器人预览使用 CPU 即可；安装了 CUDA 配置也可以运行这个命令。要测试 GPU 路径，可将 `--device cpu` 改成 `--device cuda`。

- **Payload / 负重 → 重量 (kg) → 应用并重置**：按当前模式重建质量和惯量，回到初始支撑姿态并暂停。
- **Play / Pause**：运行或暂停；**Step**：单步；**Reset Environment**：重置。
- 前臂交互预览允许箱子掉落、机器人倒地后继续模拟，不自动重置，也不计作评测结果；看完后手动 Reset。普通 smoke / 离线 demo 保留跌倒终止条件。
- 窗口中的 0–10 kg 只是仿真输入范围，**不代表真机承载额定值**。0 kg 实际用 0.001 kg，避免零惯量。
- 零动作 PD 只以设定关节姿态为目标，不是平衡控制器。播放后箱子可能滑落、机器人可能倒下；不要把暂停时的摆放当成稳定承重结果。
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
  THIRD_PARTY_NOTICES.md     依赖来源与授权边界
  RELEASE_MANIFEST.json     包内文件 SHA-256 清单
```

运行后自动生成的 `.venv/`、`outputs/` 不在分享包内。代码包不含视频、checkpoint、凭据、旧日志或第三方模型资产。依赖和机器人资产在安装时从锁定的上游提交获取；请遵守 [来源说明](THIRD_PARTY_NOTICES.md)。

需要理解“前臂接触和旧手部连接有什么不同”：看 [实物与仿真对应](docs/SIM_TO_REAL_ZH.md)。需要详细参数：看 [环境设计](docs/ENVIRONMENT_DESIGN.md) 或运行 `python -m lifting_lab COMMAND --help`（使用本项目解释器）。本包不确认真实前臂外壳、衬垫或托架可以承重。

## 6. 放到 Git 给四个人共用

建议使用一个仅小组成员可访问的独立私有仓库。把 ZIP 解压到新的文件夹，
以里面的 `lifting-lab` 作为仓库根目录，提交解压后的源码，而不是只提交 ZIP。
不要在包含其他课程作业的父目录执行 `git add .`。

- 保留 `uv.lock`、`.python-version`、`.gitignore` 和中文 README。
- `.gitignore` 已排除 `.venv/`、`outputs/`、`validation/`、日志、录像、模型权重和本地凭据文件。
- 提交前检查 `git status --short`，确认只有本项目的源码和说明；不要使用 `git add -f` 绕过忽略规则。
- 队友 clone 后进入仓库，按本文件第 1 节安装。GPU 用户选 `cu130`，CPU 用户选 `cpu`。
- 开始统一测试前对齐 `git rev-parse HEAD` 的结果，避免各自使用不同版本。
- `RELEASE_MANIFEST.json` 只对应那一次 ZIP 内容。修改代码后应重新打包生成清单，不把旧清单当成当前代码的校验结果。

只分享自己的项目源码；不上传 `.venv` 中下载的第三方实现/机器人资产，详见来源说明。
本包未自动创建远程仓库或上传文件。

## 7. 给 GPT / 队友的说明

> 请先读本 README、pyproject.toml 和 docs/ENVIRONMENT_DESIGN.md。这是独立的 K1 前臂托箱研究项目，不要修改其他课程环境。使用当前项目 .venv 的 Python。所有 CLI 默认 forearms，legacy pregrasp 须显式选择，不混用 checkpoint。先检查安装与小规模训练流程，再根据我授权的预算启动训练；不操作实体机器人，不升级依赖。不要把短测试、viewer 或高奖励当成 20 秒成功率；使用独立评测。物理/判据变化须版本化，实物对应先读 docs/SIM_TO_REAL_ZH.md。

遇到问题时提供完整命令、报错、系统和 GPU 型号。端口被占用时看终端实际打印的地址；不要随意结束别人的 Python 进程。离线、仓库访问失败或驱动不兼容时，保留错误，勿用重新生成 lock 的方式绕过。
