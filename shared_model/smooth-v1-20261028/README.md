# 平稳托箱模型 · smooth-v1 · 2026-10-28

这是可以直接回放、改重量和继续训练的研究 checkpoint。箱子初始放在两条前臂上，靠自由接触承托，没有固定连接，也不包含从地面拾取。安装方法见[仓库首页](../../README.md)。

## 队友直接使用

在仓库根目录运行，保留 `policy.pt` 与相邻的 `experiment.json`：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --checkpoint shared_model/smooth-v1-20261028/policy.pt --mass 0.5 --seconds 3600 --viewer --paused
```

打开终端打印的本地地址，按 **Play**。在 **Payload / 负重** 中输入重量，再按 **应用并重置**，然后重新播放。每次更换重量都会重置环境；viewer 的观察不等于正式评测。模型自动继承 `smooth-v1`，无需手动选择奖励。Linux 把解释器改为 `./.venv/bin/python`。

复测五种负载，需要 CUDA 安装：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab evaluate --device cuda --num-envs 50 --checkpoint shared_model/smooth-v1-20261028/policy.pt --protocol configs/smooth_hold_validation.json --output outputs/smooth_shared_validation
```

续训示例（在现有模型基础上再训练 200 次更新，不是独立的新种子实验）：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 512 --seed 11 --iterations 200 --resume shared_model/smooth-v1-20261028/policy.pt --output outputs/smooth_shared_continued
```

输出目录必须取新名字。续训继承质量、奖励及优化器等设置，环境与随机状态会重启，因此不是逐位复现原训练轨迹。从头训练和修改奖励的方法见[技术说明](../../docs/SMOOTH_HOLD_ZH.md)。

## 已有结果

同一个训练 seed **11** 的模型，评测 batch seed **5011**，每种负载 50 个 episode。目标时长 20 秒，失败可以提前终止。完整逐 episode 记录见 [evaluation.json](evaluation.json)。

| 箱子质量 | 成功数 | 成功率 |
|---|---:|---:|
| 0.247461 kg | 50/50 | 100% |
| 0.494922 kg | 50/50 | 100% |
| 0.742384 kg | 50/50 | 100% |
| 0.989845 kg | 50/50 | 100% |
| 1.484767 kg | 33/50 | 66% |

合计 233/250。最重一档的 17 次失败都触发了禁止接触判据，在约 6.84–10.68 秒终止；记录未细分是碰地还是碰到其他机器人部位。不要用总成功率掩盖该负载的失败。

这些是四个较轻离散负载点和一个压力测试，**不是整个连续质量区间、任意初态或多个独立训练 seed 的可靠性证明**。摩擦固定，未添加推力、重心偏移或箱体相对位姿扰动。原有共同初始 XY/yaw/小速度变化保留。成功判据要求满 1000 个控制步、箱体倾角小于 10°、高度偏差小于 5 cm，并满足接触及终止条件；连续失去承托仍有 6 步（0.12 秒）宽限。不要求对称站姿或零抖动。

[示例视频](hold_0p495kg.mp4) 是约 0.495 kg、seed 2027 的 20 秒回放（50 fps），不替代上面的 250 次评测。另附 CPU、同质量与 seed 的 20 秒关节诊断：0.14 秒仍有约 0.44 rad 的右膝目标跳变，2 秒后整体关节速度 RMS 为约 0.01982 rad/s。当前版本后段较平稳，**启动瞬态和不对称站姿仍待改进**。

## 模型与训练来源

- 任务 `forearms-contact-v1`；观测 `k1-load-state-97-v1`，97 维，包含真实质量比例；22 个动作。
- PPO，`smooth-v1` 奖励，直接 PD 位置目标控制；没有额外低通或动作限速。
- 固定训练质量 0.494922 kg；512 个环境，rollout 长度 24，200 + 800 次更新；合计 **12,288,000 transitions**，末次 iteration 999。
- Windows 11 / RTX 5090 Laptop GPU；依赖固定于 `uv.lock`。训练环境详情在 [experiment.json](experiment.json)。Linux 安装流程尚未在 Linux 实机验证，不同设备不保证逐位相同。
- 箱子尺寸 20 × 40 × 12 cm。箱子—前臂有效滑动摩擦系数 0.7，脚底—地面 1.0；均为仿真假设，尚未用真实材料标定。没有真机承载或部署验证。
- 训练、评测与本次发布运行源码的指纹：`b097e3415fcaa087ecc33c329d6cd34cbce7d07b43b8a0bf1cf4b986c7e61c9c`。
- 模型 SHA-256：`a65edc7fd9415035179a039a2b80aa95b3fe3eeabb0203fa1051d5833b325118`。

原始 JSON 按字节保留。里面的旧 `D:\Inventories\...\outputs\...` 或 `outputs/...` 路径用于记录训练来源，队友无需创建这些目录；实际加载请用上面的仓库相对路径。`completed.json` 是训练结束时的记录，其“not evaluated”说明不应改写；后续完成的独立评测保存在 `evaluation.json`。

## 文件清单

| 文件 | 用途 |
|---|---|
| `policy.pt` | actor、critic、归一化状态和优化器，可回放/续训 |
| `experiment.json` | 加载模型需要的相邻元数据 |
| `completed.json` | 原始训练结束记录与模型校验值 |
| `evaluation.json` | 五负载、250 个 episode 的完整结果 |
| `hold_0p495kg.mp4` | 20 秒示例视频 |
| `joint_trace.npz`、`joint_summary.json` | CPU 回放的逐步关节数据与摘要 |
| `joint_capture.json` | 该 CPU 回放的原始 `demo.json`，仅更改文件名 |
| `frequency_diagnostics.json` | 已有回放的频率和接触诊断 |
| `SHA256SUMS.json` | 本目录上述数据及本说明的字节校验值；不包含清单自身 |

源码指纹只覆盖 `src/lifting_lab/*.py`、`pyproject.toml`、`uv.lock` 的相对路径及原始字节。`.gitattributes` 保持源码 LF 换行和本目录数据原始字节，避免 Windows 克隆时改变指纹。Git commit 标识完整代码版本；文件哈希用于校验内容，不代表性能认证。
