# 空手稳站模型 · 2026-10-30

模型从零训练。场景只有机器人和地面，**没有箱子、碰撞质量代理或虚拟箱子输入**。原托箱模型保留。本目录是本地 `codex/empty-hand-stand` 分支的精选结果，尚未推送 GitHub。

## 已验证的范围

| 测试 | 结果 | 条件 |
|---|---|---|
| 普通独立 GPU 评测 | 32/32 | 两个评测 seed，每次 20 秒 |
| 小幅速度扰动 | 16/16 | 每次 22 秒；每 5 秒给 X/Y 水平速度分别增加 U[-0.05,0.05] m/s |
| CPU 回放评测 | 1/1 | 独立 seed，20 秒 |
| 零动作 PD 对照 | 0/16 | 同一空手环境，约 1.2 秒后倒下 |

普通评测稳态（2 秒后）左右脚底平均倾角约 **0.061° / 0.025°**，垂直支持力约 **99.06 / 94.99 N**，躯干平均倾角 **0.256°**，最大水平漂移 **1.27 cm**。机身高度约 **0.5142 m**，接近默认 0.5125 m；并非靠低蹲通过。平均值不是最坏时刻，完整逐 episode 记录见 JSON。

成功要求：完整时长不倒下、无非有限数；2 秒后至少 95% 帧双脚接触且各脚倾角 <5°，躯干倾角 p95 <5°，抬臂姿态误差 RMS <0.15 rad，关节速度 RMS <0.2 rad/s；全程水平漂移 <0.10 m。失败不会被重置后的新 episode 替换。接触力采样为每个 50 Hz 控制步末尾物理子步，不是所有子步的平均。

训练只有一个 seed（11）：256 并行环境 × 24 步 rollout × 300 次 PPO 更新 = **1,843,200 transitions**。不同评测 seed 不等于多个训练 seed。初始化关节扰动 ±0.015 rad，水平速度 ±0.02 m/s，位置/yaw 小幅随机；没有摩擦随机化。速度扰动也不等于测量过的真实外力推压。

## 打开 viewer

先按根目录 README 安装。在项目根目录运行（每次更换输出目录名）：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab.empty_hand_cli demo --device cpu --num-envs 1 --seed 2027 --seconds 3600 --checkpoint shared_model/empty-hand-v1-20261030/policy.pt --port 8081 --paused --output outputs/empty_hand_my_viewer
```

打开 http://127.0.0.1:8081/，点 **Play**。保留终端运行。旧托箱 viewer 可以继续使用 8080。空手 viewer 不提供箱子重量控件。

## 复测

```powershell
.\.venv\Scripts\python.exe -m lifting_lab.empty_hand_cli evaluate --device cuda --num-envs 16 --seed 8011 --seconds 20 --checkpoint shared_model/empty-hand-v1-20261030/policy.pt --output outputs/empty_hand_team_eval
```

增加 `--push-speed 0.05 --seconds 22` 可进行相同的小幅速度扰动测试。Linux 把解释器改成 `./.venv/bin/python`。详见 [空手任务说明](../../docs/EMPTY_HAND_ZH.md)。

## 文件和接口

- `policy.pt` 与相邻 `experiment.json` 必须一起保留；`completed.json` 是训练完成记录。
- Actor 为 72 维本体观测，critic 为 76 维；22 个关节位置动作、50 Hz，保留原动作缩放和默认姿态偏移。
- `evaluation_seed*.json`、`velocity_kick_seed7011.json`、`cpu_seed2027.json` 为原始评测报告；`cpu_trace.npz` 为 CPU 原始轨迹。
- `pd_baseline_seed5011.json` 为零动作对照，`selection.json` 为汇总与来源，`SHA256SUMS.json` 校验本目录文件。

这不是课程 walking ONNX，也不能替换旧 97 维托箱策略。当前从**双臂已经抬起**的姿态开始；未验证从自然下垂抬臂、接箱/卸箱或策略切换。没有真机 adapter，上机前仍须核对传感器、关节顺序/方向、执行器控制和保护逻辑。上述结果只支持已测仿真条件。
