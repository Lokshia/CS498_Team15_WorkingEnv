# 双脚平放托箱模型 · stance-v1 · 2026-10-29

此目录选用质量随机化训练的中间 checkpoint `model_1450.pt`，对外命名为 `policy.pt`。它在选型评测 seed 5011 和选择后复核 seed 6011 的三种负载上，均完成托箱和平脚判据，共 **96/96 次**。旧 `smooth-v1` 模型保留在原目录。这仍是仿真研究结果，不是广泛鲁棒性或真机部署验证。

## 打开 viewer

完成仓库首页的安装步骤后，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --checkpoint shared_model/stance-v1-20261029/policy.pt --mass 0.4949224923474894 --seed 2027 --seconds 3600 --viewer --paused
```

打开终端打印的本地网址，点 **Play**。在负重控件输入质量，点 **应用并重置**，再播放。改变质量会重置环境；viewer 画面不等于正式评测。Linux 将解释器替换为 `./.venv/bin/python`。

箱子初始已放在前臂上，靠自由接触支撑，没有固定连接；本任务不包含拿起箱子或从普通站姿进入托箱姿态。

## 已有结果与限制

GPU、每个 seed 的每个负载 16 次、每次 20 秒：

| 箱子质量 | seed 5011：托箱／平脚 | seed 6011：托箱／平脚 |
|---|---:|---:|
| 0.247461 kg | 16/16 ／ 16/16 | 16/16 ／ 16/16 |
| 0.494922 kg | 16/16 ／ 16/16 | 16/16 ／ 16/16 |
| 0.989845 kg | 16/16 ／ 16/16 | 16/16 ／ 16/16 |

平脚判据要求先满足原托箱成功，再要求完整 `t >= 2 s` 稳态窗口中至少 95% 的样本双脚同时接触、各自倾角 <5°。它不要求左右精确均分受力，也不额外限制躯干倾角。逐 episode 结果分别见 [evaluation_seed5011.json](evaluation_seed5011.json) 和 [evaluation_seed6011.json](evaluation_seed6011.json)。

约 0.495 kg、seed 2027 的匹配 CPU 回放，稳态左右脚平均倾角由旧模型的 **0.025°／10.215°** 变为 **0.019°／0.391°**；左右支持力约 **113.10／85.93 N**。但躯干平均倾角从 **2.646° 增加到 4.136°**，仍不是完全对称、端正的站姿。启动瞬态与稳态分别保存在 [stance_summary.json](stance_summary.json)。

这些结果来自同一个训练 seed 11，只有三个负载点。摩擦固定，没有推力或箱体相对位姿扰动；seed 5011 曾用于选择 checkpoint，seed 6011 是选择后的额外初态 seed 复核，二者都不是独立训练种子的重复实验。**不能据此声称任意重量都可靠，或已经能部署真机。** 97 维观测仍包含仿真箱子状态，实际测量、状态估计和控制接口尚未接通。

## 模型来源与复测

选用 checkpoint 的最后完成 iteration 是 **1450**（从零编号）：原模型 1000 次更新 + pilot300 的 300 次 + 混合质量训练的 **151 次**，累计 **17,829,888 transitions**。完整 400 次混合质量训练的末模型表现反而退步，未被选用。

- 模型 SHA-256：`b0093a17767ba382549e392dc0c2f5844dea66b37173a6b6f1ba08c9c98a18f4`
- 运行源码指纹：`3b09ef90d8b244caa2d602e0783a8bae129a916458ba02e198eba4e55b5717b8`
- 实际选用 checkpoint 的预算和来源：[selection.json](selection.json)。
- 相邻 `experiment.json` 原样保留完整 mixed400 run 的配置和结束记录，用于兼容加载。`original_training_completed.json` 属于该 run 的后续末模型，**不是当前 policy.pt 的完成证书**。请勿用其中末模型哈希或 400 次预算替代 selection.json。

复测选型条件：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_stance.py --checkpoint shared_model/stance-v1-20261029/policy.pt --device cuda --num-envs 16 --seed 5011 --seconds 20 --mass 0.247461246173745 --mass 0.494922492347489 --mass 0.989844984694979 --output outputs/stance_shared_recheck
```

输出目录必须为新目录。要复测第二组，将 `--seed 5011` 改为 `--seed 6011` 并换一个输出目录。完整奖励和训练说明见 [STANCE_HOLD_ZH.md](../../docs/STANCE_HOLD_ZH.md)。

## 文件清单

| 文件 | 用途 |
|---|---|
| `policy.pt`、`experiment.json` | 模型和相邻兼容配置，需一起保留 |
| `selection.json` | 当前 checkpoint 的实际 iteration、预算、来源和选型状态 |
| `original_training_completed.json` | 原始完整 mixed400 run 的结束记录；不对应当前模型 |
| `evaluation_seed5011.json` | 当前模型的匹配批量评测，含所有 episode 结果 |
| `evaluation_seed6011.json` | 模型选择后的第二个评测 seed 复核，含所有 episode 结果 |
| `stance_summary.json`、`stance_trace.npz` | 当前模型单次 CPU 姿态／受力诊断及逐步数据 |
| `baseline_*` | 同条件旧 smooth-v1 的诊断与批量评测 |
| `pilot300_*` | 首轮候选的诊断与批量评测，保留约 1 kg 启动失败 |
| `rejected_mixed400_evaluation_seed5011.json` | 未选用的最终 mixed400 模型结果，避免只报告较好候选 |
| `SHA256SUMS.json` | 本目录其他文件的 SHA-256 校验清单，不含清单自身 |

JSON 按原始字节保留，内部绝对路径用于记录来源，不要求队友创建同名目录。汇总 JSON 已含逐 episode 结果；其中提到的批量原始 NPZ 未随包复制，只有当前模型的单次 CPU trace 包含在本目录。
