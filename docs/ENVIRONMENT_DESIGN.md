# 环境设计：v0.4.0 前臂托箱训练与 legacy pregrasp

## 默认仿真、训练和评测：forearms

`demo` / `smoke` / `train` / `evaluate` 默认使用 `--support-mode forearms`，也可明确指定
`--support-mode pregrasp` 对照旧模式。两者均可选择 CPU 或 CUDA。
先用 `demo --viewer --paused` 观察初始接触，再单步或播放；暂停姿态不是
静态平衡求解结果，更不是训练成果。

- 两条机器人前臂以近水平姿态支撑箱底，使用机器人实际前臂碰撞几何，
  不是在手下放置一个固定在世界坐标中的隐形平台。
- 箱子是独立六自由度刚体。forearms 不添加箱子与手/身体之间的
  `CONNECT`、`WELD` 或其他 equality 连接，也不添加旧版手/箱碰撞排除。
  重力、接触法向力和摩擦决定运动；允许滑动、分离、倾翻和掉落。
- 机器人基座自由，22 个关节仍由执行器控制。零动作 PD 仅跟踪初始关节
  姿态，不会自动选择修正动作来平衡负载。接触模型存在不代表箱子一定稳住。
- 重量控件按当前模式重新创建物理模型，质量与惯量一起更新，重置后暂停。
- `forearms` 交互 viewer 关闭跌倒终止和自动重置，以便观察完整掉落；非有限状态检查仍保留。普通 smoke / 离线 demo 仍有跌倒终止条件。预览不是正式评测。
  请求质量低于 0.001 kg 时仍采用 1 g 数值下限；UI 的 0–10 kg 不是硬件额定值。
- 没有外部 push、抓握学习、训练成功的负重平衡策略或真机验证。接触几何和摩擦参数
  是仿真配置，不是测得的真实前臂外壳/衬垫参数。

前臂训练与评测使用 `forearms-contact-v1`，观测 schema 为 `k1-load-state-97-v1`。
97 维观测顺序沿用下表，但 88–93 项只是箱体相对手臂参考点的几何偏差，不是
抓握残差或真实接触力。观测不增加接触力/接触计时输入，质量盲仅遮蔽质量项。
因此接触与宽限计时对策略并非完全可观测；这是基线限制，未来增加传感或历史
观测应更新 observation schema，并重新训练，而不是直接载入旧 checkpoint。
22 个动作控制自由机器人关节，没有固定基座、固定手臂或姿态模仿奖励。

真实接触传感器单独用于奖励、诊断和失败判断。只有左右指定前臂圆柱计有效承托；
两侧末端球允许辅助接触，但不能替代圆柱承托。物体接触地面或其余机器人
碰撞体判失败。双侧圆柱接触且分别有 >1e-6 N 世界向上力
才为 supported；连续不满足超过 6 控制步（0.12 s）失败，启动不额外叠加宽限。
每个 control step 提交一次接触快照和计数，读取奖励/终止/指标不重复推进历史；
partial reset 仅重置指定环境行。接触快照与 pinned mjlab 训练信号对齐，来自
末物理子步，最多存在一个 0.005 s 子步的采样时差。

MuJoCo-Warp 对 cylinder–box 等凸形状对目前最多生成一个接触点，上游启动时
会提示 MULTICCD 限制。这里不把这种近似当成真实面接触/接触压强标定。
底层也可能打印 ground friction 数值下限提示；不要据此忽略后续 NaN 或训练
不稳定。软件有限值检查不能替代摩擦、碰撞几何和真实承重方式的验证。

高度、倾角、机器人直立、运动与动作平滑等既有 reward 作为初始设置，另加
前臂承托奖励；具体权重以 `env_cfg.py` 为准。这些是待研究/调参的配置，不是
已验证能收敛的配方。严格成功率由独立 evaluator 计算，不根据累计 reward 判断。
完整判据见 [团队协议](TEAM_PROTOCOL.md) 和 `configs/team_protocol.json`。

训练支持单质量、顺序课程、质量盲对照和继续训练；checkpoint 校验模式、任务
schema、观测 schema 与质量观测变体。旧无 schema 权重只作为显式 pregrasp
兼容路径。两个模式不混载。前臂 checkpoint 可在 viewer 回放，但交互预览关闭
任务失败终止以观察完整掉落，不会生成正式成功率；离线 demo/评测保留判据。

`scripts/check_training.py` 是有界流程验收，不是学习性能测试。旧 v0.2.0/v0.3.0
ZIP 保持不变；团队必须重新解压 v0.4.0 或同步到新 commit。

### 可选随机化：按环境的质量/重心、箱子位姿抖动、推力

`make_env_cfg` 的这些参数默认关闭，关闭时编译出的模型、事件和观测与上文完全一致：

- **按环境质量与重心**（`payload_mass_range_kg`、`payload_com_offset_m`）：
  一个 `mode="reset"` 事件 `mdp.randomize_payload_inertial` 对被 reset 的环境
  直接写 Warp 模型的 `body_mass` / `body_ipos` / `body_inertia`（`payload/box`
  那一行）。事件用 `@requires_model_fields(..., recompute=set_const)` 声明，
  mjlab 会在建环境时把这些字段及其派生量（`body_subtreemass`、
  `*_invweight0` 等）扩展成每个 world 一份，并在事件触发后调用
  `mjwarp.set_const` 重算派生常量（保存并恢复 qpos，不改状态）。质量在区间内
  均匀抽样，低于 1 g 仍取 1 g；惯量始终是"以新重心为中心、按抽样质量缩放的
  均匀箱惯量"，即把偏心配重近似为重心平移而不改变主轴。每次有 reset 事件触发
  都会对全部 world 重算一次常量，开销大致相当于一次 forward；用 reset 而不是
  step 模式就是为了控制这个成本。
- 观测第 94 项改为从 Warp 模型读该环境的实际箱体质量（无随机化时就是编译值，
  数值不变），`mass_blind` 仍置零。重心偏移**不**进入观测：mass-conditioned
  策略知道重量但不知道配重在哪，mass-blind 两者都要从动力学推断。
- **箱子位姿抖动**（`payload_xy_jitter_m`、`payload_yaw_jitter_rad`）在
  `reset_pregrasp` 里只作用于箱子，在共同 yaw 之前叠加：x 只向前取 `[0, j]`，
  因为名义位置的箱子后沿距躯干碰撞体只有 1 cm；y、yaw 对称。yaw 后箱底最多比名义
  静止高度高 0.3 mm，所以统一抬 0.2 mm 起步。用 64 环境 × 3 次 reset 的零动作
  测试确定边界：1 cm / 0.03 rad 以内所有起步都在 3 步内获得双圆柱承托；2 cm 时
  13–20% 的起步落在末端球上，6 步内拿不到承托。若同时把重心横向偏移到 4 cm 以上，
  仍有 5–10% 的起步只压住一侧圆柱，这是 v1 判据（必须双圆柱受力）与
  单接触点近似共同造成的，训练时请自行权衡偏移幅度。
- **推力**（`push_interval_s`、`push_velocity_m_s`）直接使用 mjlab 的
  `push_by_setting_velocity`：`mode="interval"`，每个环境独立计时、reset 时重抽，
  触发时给机器人根连杆的世界系 x/y 速度各加一个 ±V 的均匀增量。这是瞬时、与质量
  无关的速度踢，不是力；箱子不被踢，只通过前臂接触跟随。按固定水平面上的滑块
  粗略估算，1 cm 停止距离与摩擦系数 0.7 对应 √(2·0.7·9.81·0.01) ≈ 0.37 m/s。
  这忽略了机器人运动、箱体转动及瞬态接触，不能视为当前系统的硬速度上限或
  必然碰撞条件。协议采用 0.3 m/s 是待验证的设定，实际表现须由仿真和后续标定判断。
- 这些都只改变训练分布/评测条件，不改观测、动作或成功判据，`task_schema`
  保持 `forearms-contact-v1`；实际使用的范围写在 `experiment.json` 的
  `randomization` 与 `mass.mass_range_fraction`，续训默认沿用。带推力评测用
  `configs/team_protocol_push.json`，结果单独报告。

## Legacy pregrasp：以下原设计仅用于旧模式

This retained design describes **holding an attached load at chest height
while balancing**. It does not implement ground pickup, grasp acquisition, or a
real-robot controller. A successful software smoke test is not evidence that a
trained policy meets the task's success criterion.

### Physical model (legacy pregrasp)

- The pinned Booster K1 has a free-floating base and 22 actuated joints. Its
  unloaded model mass is approximately **19.7969 kg**. The base is not fixed to
  the world; corrective stepping is allowed.
- The payload is a separate free body, not extra torso mass. The box dimensions
  are 0.20 × 0.40 × 0.12 m. Its mass and homogeneous-box inertia both change with
  the experiment's mass. For half-sizes `(a,b,c)`, the diagonal inertia is
  `m/3 × (b²+c², a²+c², a²+b²)`. Handles are massless visual geometry.
- Requested **0 g is simulated as 1 g**, because a free body needs positive
  inertia. The same floor applies to any requested mass below 0.001 kg. Both
  requested and actual mass are recorded; the mass-conditioned observation uses
  the actual mass. The zero-load condition is therefore a numerical proxy.
- Two compliant MuJoCo **CONNECT** constraints join the wrist sites to the
  payload's handle sites. Exact forward kinematics aligns both connections at
  reset. The initial box-center height is approximately **0.610 m** and the
  hands are approximately 0.640 m above the floor.
- Two point connections do **not** make a full rigid weld: rotation about the
  line through the two grip points remains possible. The box center is 3 cm
  below that line. There is no finger model, grip-force limit, or force-triggered
  release. The 0.15 m connection-error termination is a numerical/geometric
  failure check, not a physical grip-strength threshold.
- Gravity is `(0,0,-9.81)` m/s². Robot/ground, payload/ground, and robot self
  collisions remain active; secured-hand/payload collision pairs are excluded.
  The original motor parameters and torque limits remain. Stochastic actuator
  delays are disabled in this baseline.

Each batch has one fixed payload mass. A mass sweep constructs and compiles a
fresh environment per condition, avoiding stale mass/inertia caches. CPU and
CUDA use the same task configuration. Physics runs at 200 Hz; actions at 50 Hz.

### Actions and observations (legacy pregrasp)

The 22 actions are joint-position offsets around the initial holding pose,
using the pinned K1's per-joint action scales. CLI training and checkpoint
inference clip actions to `[-1,1]`. There is no reward that requires joints to
remain at that initial pose.

Actor and critic both receive **97 features**, with no observation noise:

| Zero-based indices | Features |
|---|---|
| 0–74 | Base angular velocity (3), projected gravity (3), relative joint positions (22), joint velocities (22), previous actions (22), base linear velocity (3) |
| 75–86 | Payload position relative to robot (3), relative linear velocity (3), payload-frame projected gravity (3), payload angular velocity expressed in the robot frame (3) |
| 87 | Signed payload height error |
| 88–93 | Left/right grip displacement vectors in the robot frame |
| 94 | Actual payload mass / unloaded robot mass |
| 95–96 | Robot base height and payload center height |

**Mass-blind** masks only index 94 to zero, for both actor and critic. It keeps
the same dimensions and physical mass. Other state features still permit
implicit mass inference; “blind” does not mean the dynamics hide the load.

### Reward and diagnostics (legacy pregrasp)

The weighted reward terms are height tracking (+3), level payload (+2), upright
robot (+2), approximate COM margin (+0.5), joint effort (−0.00002), action changes
(−0.02), payload motion (−0.05), contact-gated foot slip (−0.2), and soft joint
limits (−0.1). Rewards are scaled by the 0.02 s control interval. These weights
are a starting configuration, not a claim of optimal tuning.

Combined COM uses **world inertial centers (`xipos`) weighted by actual body
mass**, including the payload. `com_margin_approx_m` is the signed margin inside
an **axis-aligned envelope of both sole centers**, extended by fixed half-widths
0.07 m and 0.025 m. It is not the actual contact polygon, center of pressure,
ZMP, or a dynamic stability guarantee. It can overestimate support when a foot
is airborne or rotated. Its reward is disabled when neither foot contacts the
floor. Foot-slip cost uses squared horizontal sole velocity only for contacting
feet.

The fixed protocol requires a complete **20 s / 1,000-control-step episode**
with no failure, payload tilt strictly below 10°, and absolute height error
strictly below 0.05 m at every recorded step. There is no settling grace period;
averages cannot replace these maximum-error tests. Corrective steps are allowed.
`valid_hold` is only an instantaneous diagnostic, not episode success.

Failures include robot base below 0.28 m, robot tilt above 60°, payload center
below 0.20 m, excessive connection error, or non-finite state. A normal time
limit is not a fall. Reports also retain torque RMS, contact count, slip, grip
error, heights, and tilt; a zero-action PD demonstration is labelled untrained.

### Fair resets and team comparisons (legacy pregrasp)

Seeded reset variations apply a **common** XY translation (±0.05 m), yaw
(±0.05 rad), and XY velocity (±0.02 m/s per component) to robot and payload.
This preserves the pregrasp; joints are not independently jittered. All three
ranges can be set to zero through `make_env_cfg` for identical resets. There
are no external pushes or startup domain randomization in V1.

Construct the policy/wrappers before the final seeded evaluation reset, so an
implicit wrapper reset cannot change one baseline's initial state. Failed
episodes remain failures even if their simulator slots are reset while other
slots finish. Compare identical protocol, seeds, batch size, source fingerprint,
and runtime; cross-device bitwise equality is not promised.

The protocol trains at requested mass fractions 0%, 2.5%, and 5%; evaluates at
1.25%, 3.75%, and 7.5% (stress condition); and uses 50 episodes per seed/condition.
Quick tests do not satisfy that evaluation. Sequential fixed-mass curriculum
phases are not simultaneous per-environment mass randomization.

## Permission and scope for both modes

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Dependency locks identify
versions, not redistribution rights. This project does not grant a license to
third-party robot meshes, XML, or implementation code. The intended source
archive excludes those assets, environments, caches, and trained weights;
obtain the appropriate upstream/course permission before redistributing them.

Neither the forearms exploration nor legacy pregrasp provides a push-recovery
result, pickup result, finger-grasp result, or
real-robot safety endorsement. Software tests and short rollouts establish only
the properties they actually check. Policy quality must be measured separately
under the full protocol before reporting a success rate.
