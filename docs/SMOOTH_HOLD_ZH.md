# 平稳托箱：问题、训练设置与诊断

当前方案是 **`smooth-v1` 奖励 + 原有直接位置目标控制 + 97 维观测**，在 forearms 自由接触任务中减少颠箱、腾空和快速蹲伸。箱体、质量/惯量、摩擦、碰撞、PD、动作尺度及接触成功判据保持原定义。没有启用目标滤波或限速模块。

本分支附带 [已训练模型和精选证据](../shared_model/smooth-v1-20261028/README.md)。直接运行或续训的命令见 [根 README](../README.md)。`policy.pt` 必须与其相邻 `experiment.json` 一起使用；这里列出的历史本地诊断不等于全部已发布。

## 已确认的问题

组员提供的七条回放中，基座高度谱峰约 **7.55–8.20 Hz**，双前臂有效支持比例约 **29.9%–35.2%**，最长失载常到 5 个控制步后恢复。分析汇总见 [频率与支持诊断](../shared_model/smooth-v1-20261028/frequency_diagnostics.json)；原始组员回放未随本分支发布。旧判据允许连续失载最多六步，单次恢复会清零计数；因此通过旧判据不等于一直稳稳托住。

我们没有拿到组员原始 checkpoint，**没有复现其确切权重和完整训练过程**。以下关节数据来自本地另行训练的 smooth 200-update pilot，不应混称为组员模型：右膝 raw action 相邻差 RMS 为 **0.733850**，对应位置目标相邻差 RMS 为 **0.256848 rad / 20 ms**，关节速度 RMS 为 **2.7380 rad/s**。该段右膝目标仍在硬限位内；目标跳变是真实控制输入中的快速激励，不是视频播放效果。

四秒 CPU 零动作诊断没有复现持续跳跃，而是逐渐前倒。诊断关闭了 termination/自动 reset，不能作为成功成绩。FK 检查表明，小幅肘关节变化可以让水平箱底同时接近两圆柱与辅助端球，但这不是完整受载平衡证明。该零动作/FK 诊断及上述 200-update pilot 关节数据仅为历史本地记录，未随本分支发布；本地审计文件为 `outputs/smooth_study_20261028/control_structure_cpu/AUDIT_ZH.md`。

## 当前奖励：`smooth-v1`

`--reward-profile baseline` 保留原奖励；`smooth-v1` 仅用于 forearms。保留高度、箱水平、躯干直立、近似平衡、扭矩和支持奖励，调整或新增：

| 项 | smooth-v1 权重 | 含义 |
|---|---:|---|
| raw action 相邻差平方和 | −0.15 | 原值 −0.02 |
| 箱体运动成本 | −0.2 | 原值 −0.05；线速度平方和加 0.1 倍角速度平方和 |
| 接触脚滑动成本 | −0.5 | 原值 −0.2 |
| 未获得双侧有效承托 | −2 | 每个失载样本计成本，宽限内也计 |
| 缺少接触的脚数 | −2 | 两脚接触为 0，单脚为 1，腾空为 2 |
| 躯干竖直速度平方 | −4 | 抑制上下运动 |
| 躯干角速度平方和 | −0.2 | 抑制快速转动 |
| 关节速度平方和 | −0.002 | 抑制快速关节运动 |

这是训练 shaping，不增加新的成功阈值。单脚纠正步仍可进行，不会仅因缺一只脚接触就终止。

## 控制与 checkpoint

actor/critic 输入均为 **97 维**。每步采用 `target = default_joint_position + action_scale * raw_action`，训练和推理均将 raw action 截到 `[-1,1]`。原生 PD 使用该目标；当前没有额外低通、目标变化率限制或硬限位目标裁剪。

checkpoint 旁的 `experiment.json` 保存 reward profile、观测定义和训练配置。未显式指定时，回放、评测和续训继承 checkpoint 的 reward profile；demo/evaluate 若显式指定冲突的 profile，会拒绝运行。训练 resume 可以显式切换 reward profile，并记录变化。复制实验时应保留 checkpoint 及其同目录的 metadata。

做奖励对照时应保持质量、seed、环境步数和评测条件一致。不能仅比较 200 与 1000 update 的结果，就把全部提升归因于某项奖励。

## 已完成的初步验证

本地 `smooth-v1` 先训练 200 update，再续训 800 update；512 个环境、每轮 24 步，累计 **12,288,000 环境步**。在固定载荷比例 0.025（约 0.49492249 kg）下，用调参时已查看过的 **batch seed 1011、8 个 episode、每个 20 s** 得到：

| 指标 | 初步结果 |
|---|---:|
| 完整 episode 成功 | 8/8 |
| 双侧前臂有效支持比例 | 98.15% |
| 双脚均无接触的采样比例 | 0% |
| 逐步关节速度 RMS 的平均值 | 0.05247 rad/s |
| 逐步 raw action 差 RMS 的平均值 | 0.01193 |
| 逐步关节扭矩 RMS 的平均值 | 约 2.18 Nm |

这次 8-episode 调参评测仅为历史本地记录，未随本分支发布（原文件 `outputs/smooth_study_20261028/smooth_1000_eval/evaluation.json`）。这里的 8 个 episode 不是 8 个独立训练 seed。下面的五负载复测附有发布的完整结果，使用另外的评测 seed；以上及以下结果均没有使用限速控制器。

## 五种负载的完整时长复测

对同一个 **训练 seed 11** 的 checkpoint，按 `configs/smooth_hold_validation.json` 使用未用于上述调参的 **评测 batch seed 5011**，每种负载均为 **50 个并行环境、50 个 episode**。每个 episode 目标时长 20 s，失败可提前终止。没有启用推力、COM 偏移或箱体相对位姿 jitter；原有共同 XY/yaw/小速度 reset 变化保留。GPU 结果：

| 实际载荷 | 成功 episode | 成功率 | 用途 |
|---|---:|---:|---|
| 0.247461 kg | 50/50 | 100% | 较轻负载 |
| 0.494922 kg | 50/50 | 100% | 训练负载 |
| 0.742384 kg | 50/50 | 100% | 较重负载 |
| 0.989845 kg | 50/50 | 100% | 较重负载 |
| 1.484767 kg | 33/50 | 66% | stress 测试 |
| 合计 | 233/250 | 93.2% | 包含 stress |

证据：[五负载 evaluation.json](../shared_model/smooth-v1-20261028/evaluation.json)、[训练配置 experiment.json](../shared_model/smooth-v1-20261028/experiment.json) 和 [训练完成记录 completed.json](../shared_model/smooth-v1-20261028/completed.json)。当前验证范围可描述为**约 0.25–1 kg 的四个离散负载点**；不能推断整个连续区间或任意初态都可靠，也不是多个独立训练 seed 的重复验证。

1.48 kg 的 17 个失败 episode 都记录了禁止接触，在约 6.84–10.68 s 提前终止。该高负载下仍不能可靠维持允许的支撑方式。当前 JSON 合并了箱体与地面/其他机器人部位的禁止接触，不能仅据此点名“碰胸”或断言实际跌倒；也不能用整体 93.2% 掩盖 stress 条件的 66%。

### 启动瞬态仍然存在

同一 checkpoint 在 **CPU、seed 2027、约 0.495 kg、20 s** 的独立关节诊断中，0.14 s 时右膝相邻位置目标仍跳变约 **0.43988 rad**，0.16 s 时头部 yaw 速度达到 **4.72991 rad/s**。2 s 后，全部关节及时间样本合并计算的速度 RMS 为 **0.01982 rad/s**，说明后段明显平稳，但不能描述为“全程无抖动”。证据为 [原始关节轨迹](../shared_model/smooth-v1-20261028/joint_trace.npz)、[关节汇总](../shared_model/smooth-v1-20261028/joint_summary.json) 和 [捕获记录](../shared_model/smooth-v1-20261028/joint_capture.json)（原名 `demo.json`）。[约 0.495 kg 演示视频](../shared_model/smooth-v1-20261028/hold_0p495kg.mp4) 便于观察动作，不替代数值评测。

这里的后段整体 RMS 与上表初步验证中的“逐步 RMS 再取平均”统计方式不同，不应直接相减比较。低平均速度不会消除启动峰值；仿真成功也不是可直接用于实机的安全证明。

## 从头训练示例

在项目根目录、已配置 GPU 的 `.venv` 中运行。以下是固定质量、单训练 seed 的 1000-update 起点；不加 `--resume` 或 `--curriculum`。不保证每次训练都得到相同表现。

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --support-mode forearms --reward-profile smooth-v1 --mass-fraction 0.025 --num-envs 512 --iterations 1000 --rollout-steps 24 --seed 11 --output outputs/smooth_fixed025_s11_1000
```

训练预算为 12,288,000 环境步。没有启用推力、箱体位置/COM 或质量区间随机化；原有共同初始 XY/yaw/小速度变化仍保留。输出目录必须是新目录。训练完成只说明循环及保存结束，不说明策略已稳定。

也可分成 200 + 800 update；第二条命令继承第一阶段的质量、reward profile、rollout 长度和优化器状态：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --support-mode forearms --reward-profile smooth-v1 --mass-fraction 0.025 --num-envs 512 --iterations 200 --rollout-steps 24 --seed 11 --output outputs/smooth_fixed025_s11_200
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --support-mode forearms --resume outputs/smooth_fixed025_s11_200/policy.pt --num-envs 512 --iterations 800 --seed 11 --output outputs/smooth_fixed025_s11_continued1000
```

续训会重新启动环境和随机状态，因此不是一次连续运行的逐位复现。下面评测和诊断示例使用一次训练的目录；分阶段训练时改为最后续训目录。

复现上述五负载评测的命令示例，使用 checkpoint 自动继承的配置及显式验证协议；训练 seed 11 会对应评测 seed 5011：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab evaluate --checkpoint outputs/smooth_fixed025_s11_1000/policy.pt --protocol configs/smooth_hold_validation.json --device cuda --num-envs 50 --output outputs/smooth_fixed025_s11_1000_eval
```

**训练结果看 `outputs` 内对应的 `evaluation.json`，本文不预先承诺收敛。** `--quick` 是短检查，不能替代完整协议的评测。

## 诊断关节与回放

对自己或可信组员的 checkpoint 捕获一个 CPU episode；下面命令不录视频、不训练。发生终止时停止，不偷偷 reset 拼接成完整时长：

```powershell
.\.venv\Scripts\python.exe scripts/capture_joint_trace.py --checkpoint outputs/smooth_fixed025_s11_1000/policy.pt --device cpu --mass 0.4949224923474894 --seconds 20 --seed 1011 --output outputs/smooth_fixed025_joint_trace
.\.venv\Scripts\python.exe scripts/analyze_hold_trace.py outputs/smooth_fixed025_joint_trace/demo.json --output outputs/smooth_fixed025_trace_summary.json
```

`capture_joint_trace.py` 生成 `joint_trace.npz`、`joint_summary.json` 和 `demo.json`。`analyze_hold_trace.py` 只分析已有 JSON，也可传入含多个 `demo.json` 的目录。它不加载模型、不使用 GPU。所有输出均拒绝覆盖已有文件。

读结果时分开看：

- `action_delta_rms_per_step`：raw policy 指令的相邻差，无量纲。
- `joint_pos_target_delta_rms_rad_per_step`：实际写入的位置目标相邻差，单位 rad/控制步；当前直接控制下按各关节动作尺度映射。
- `joint_velocity_rms_rad_s`：仿真中真正发生的关节运动，单位 rad/s。
- 支持/腾空比例、连续失载长度、箱体高度/倾角与终止原因：是否只是在颠箱，以及是否早早失败。

raw 动作差与实际目标差的单位、关节尺度不同，不能互相冒充。目标变化较小也不能单独证明关节或箱体稳定。相邻差汇总只使用已记录的连续 post-step 样本，不包含 reset 到第一帧；子步尖峰和采样混叠仍未被完整测量。

若捕获提前失败，即使平均速度低、画面短暂平静，也不能称为稳定托箱。少于四秒的回放不报告频谱峰；较长回放的频谱也只是描述，不单独证明振荡成因。

## 未启用的备选原型

本地 `outputs/smooth_study_20261028/proposed_control/` 留有 `limited-v1` 历史研究原型，未随本分支发布：时间常数 0.06 s 的目标低通加 2 rad/s 目标限速，并追加 22 维控制器状态，输入合计 119 维。**它没有合入当前运行源码，也没有用于上面的训练或成绩；当前 CLI 不提供 `--control-profile`。** 旧 97 维 checkpoint 不能直接用于该 119 维原型。目标限速只约束指令变化，不是实际关节速度或实机安全保证。
