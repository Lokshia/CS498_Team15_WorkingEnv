# 给 TA 的新版托箱交接：2026-11-16 / phase22 final3450

本包用于 **Booster K1 的原版 booster_deploy 新增 task**，任务名 `k1_team15_carry`。
沿用现场 walking 的 Python 3.10 / ROS Humble / aarch64 环境与官方控制器。
对照上游提交为 `cecc00fc019de3c640af2d0421059b59a2e080c1`。
不是替换 walking 的单个 ONNX，也不使用另一个 hardware bridge。

本次模型 ONNX SHA256：
`66a259aa480bd6eb6f4fc9667ebdb3bbf1464b85f805e4cf23edcb7623edadb4`。
对应训练 checkpoint：
`a2451e547f6bb232b36e289c4dea478fa11d23322fa44a7f0ff1bfbf67fabd10`。

## 可以直接发给 TA 的英文

Hi, this is our updated no-vision box-carrying task for the Booster K1.
It is registered as `k1_team15_carry` and runs through the original
`booster_deploy/scripts/deploy.py`, using the existing robot controller.
The ZIP includes the task code, ONNX model, matching control contract,
test vectors, offline checks, ROS bag recording script, and simulation results.

The policy uses proprioception and velocity commands only. It does not need a
camera, motion capture, box tracking, or the training environment on the robot.
We improved the posture after stopping while retaining slow carrying and recovery
from simulated pushes and downward pressure. The current validation scope is a
0.5 kg box and 0.1 m/s forward motion. All 15 declared native simulation trials
and 3 empty-hand standing trials passed; additional GPU and late-disturbance
tests are reported separately. This updated model has not been tested on hardware.

Could we first run the offline/read-only checks in your working deployment
environment, then do a supported zero-command test before placing the box and
trying slow forward motion? Please use the full task folder and its three model
files together. We would also like to record `/low_state`, `/joint_ctrl`,
`/remote_controller_state`, and `/rosout` for analysis.

## 文件和操作

先读包根目录 `START_HERE_ZH.md`，现场完整命令在 `README_ZH.md`。

1. 解压整个 ZIP。安装器将 `tasks/team15_carry/` 放入 TA 的已有原版仓库。
   `policy.onnx`、`policy.contract.json`、`policy.test_vectors.npz` 必须为同一套。
   `.pt`、Windows `.venv`、训练输出和教师网络都不需要部署。
2. 如 TA 仓库已有不同版本的 `tasks/team15_carry/`，安装器会拒绝覆盖。
   停止运行后，将旧 task 整个备份到仓库 `tasks` 目录以外，再装新包。
3. 安装、`deploy.py --list`、task check、只读 preflight 都不控制电机。
   Linux ARM64 上的依赖、推理时延、ROS 接收和官方控制流程仍须现场检查。
4. TA 支撑并掌握停止流程后，再启动 deploy。**X/x 准备姿态时机器人就会运动**；
   A/r 才运行策略。先零速度、空手站立；放好0.5 kg箱子后才缓慢尝试前进。
   模型不会自行拿箱；抬臂、放箱和模式切换没有被本次仿真覆盖。
5. 空格仅清零速度请求，不是急停。Ctrl+C退出 damping 不保证站立。
   实体急停、支撑和唯一控制权采用 TA 已验证的流程。
6. 官方键盘速度显示为归一化输入，不能把一次 `w` 当作0.1 m/s。
   通过策略 JSONL 的 requested/applied command 核对实际指令。
7. 录制完整 ROS bag 和策略 JSONL，同时保存安装回执与 preflight JSON。
   不在机器人网络回放含 `/joint_ctrl` 的 bag。

## 测试与仍未解决的部分

- 固定原协议 `walking_recovery_window_v3`，5场景×3个对照seed、每条20秒：15/15。
- 真正移除箱子的零命令站立：3/3。空手行走不在当前资格范围。
- GPU8方向组合：8/8；本地历史未用过的seed91827三场：3/3。
- 35秒额外诊断：22秒受推、28秒受压，恢复通过；与原20秒验收分开。
- 同seed三条combined停步窗口倾角p95平均9.07°→5.09°；右脚支撑78.06%→68.88%。
  原15条停止窗口两脚底倾角p95均≤0.02°。
- 仍约5°侧倾；晚压后右脚支撑约74.6%。推后高频腿部运动仍高于旧基线。
  不能说完全对称、无抖动或已证明真机鲁棒。
- 箱体0.20×0.40×0.12 m；0.5 kg；自由刚体，无焊接，允许轻靠胸部且双臂承托。
  箱子—前臂有效滑动摩擦0.7、脚底—地面1.0，尚未实物标定。
- 8 N侧推、3 N下压只是仿真测试条件，不是真机施力指导或额定承载能力。
  下压力不等于箱体质量变化；没有验证3–5 kg。

本包不包含前期失败候选的权重；它们保留在开发机实验记录中。新模型完整验收轨迹
和补充结果随包保留，不能将不同种类的测试次数相加成总体成功率。
