# 原版 booster_deploy：Team15 无视觉托箱 task

这是供现有 `booster_deploy` 使用的 **task 插件**。安装后仍通过原版 `scripts/deploy.py` 启动，使用原来的 ROS 控制器、遥控器和模式切换。无需运行 `hardware_bridge/run.py`，也不需要填写那份含 null 的独立控制器配置。

任务名：`k1_team15_hold`。模型输入为 IMU 角速度/重力方向、22 个关节的位置和速度、上一动作，共 72 维；输出 22 个关节目标。无需相机、箱子位置或滑移检测。

**状态：模型和 task 可以离线检查；真实机器人尚未验证。** 新 task 保留已有托箱模型与运行参数。`X/x` 的抬臂准备动作和准备增益到策略增益的切换仍需现场验证，不能用带箱仿真结果代替。

## 1. 安装到现有环境

使用机器人上原来能运行 walking 的 Python 和 ROS 环境。不重装 ROS、不升级 Torch，不复制 Windows 虚拟环境。

完整解压这个包。在本目录运行，路径改成现场现有 `booster_deploy` 目录：

```bash
python3 install.py --deploy-root ~/booster_deploy
cd ~/booster_deploy
python3 scripts/deploy.py --list
python3 -m tasks.team15_hold.check
```

列表应出现 `k1_team15_hold`；离线检查验证模型、534 条测试向量、配置和关节映射。这里不会连接 ROS 或发送电机指令。安装器只添加 `tasks/team15_hold/`，不覆盖官方文件或 walking task；同名目录已存在时停止，不会覆盖队友的修改。

对照的官方版本是 `7bb1462e2ac742eba5c01659b815e2e99fbe617f`。安装回执会记录现场版本和官方关键文件 hash。不同版本会提示，是否兼容仍看实际检查结果。

## 2. 先开数据录制

另开一个已加载相同 ROS 环境的终端，在**本包目录**运行：

```bash
bash scripts/record_rosbag.sh
```

会打印新 bag 目录。它记录 `/low_state`、`/joint_ctrl`、`/remote_controller_state`、`/rosout`，不启动策略。先保持这个终端录制。

详细字段、QoS、结束和检查方法见 [录制说明](docs/RECORDING_ZH.md)。ROS bag 没有足底接触力或箱子位置；`tau_est` 是估计关节力矩。不要在连接真机的网络中回放含 `/joint_ctrl` 的 bag，否则可能重发动作。

## 3. 现场运行原版 deploy.py

回到控制终端，在现有 `booster_deploy` 根目录：

```bash
export TEAM15_LOG_DIR="$PWD/logs/team15_policy"
python3 scripts/deploy.py --task k1_team15_hold --device cpu --exit-mode damping
```

这是实际控制入口，不是只读检查。按键沿用原版：

1. **X / 键盘 x**：进入 Custom，用官方准备阶段增益，在约 1 秒内插值到本 task 的抬臂/屈膝参考姿态。这一步已经会运动。
2. **A / 键盘 r**：启动托箱策略。策略在开始时检查当前姿态，但不能保护前一步 X 的准备过程。
3. **Ctrl+C**：请求结束，按命令选择退出到 damping。damping 不能当作自动站稳，需 TA 提供支撑；若实验室采用其他退出方案，沿用现场已确认的方式。

原版准备循环内没有逐步的中止检查，RPC 也没有这里另加的超时；因此 X 阶段不能把 Ctrl+C 当作可靠即时急停。首轮由 TA 使用实体停止和支撑设施。是否空手启动、何时放箱，由现场人员控制；程序不自动识别箱子或执行拾取。

## 4. 保存试验数据

策略会在 `TEAM15_LOG_DIR` 中新建 JSONL，保存模型/配置标识、时间、观测、网络动作和返回的关节目标。它表示策略计算结果；真正发布的指令看 bag 中的 `/joint_ctrl`，两者都不是硬件执行回执。

控制结束后，回到 bag 终端按 Ctrl+C，等待录制程序退出，然后：

```bash
ros2 bag info <刚才打印的bag目录>
```

保留整个 bag 目录（包括 `metadata.yaml` 和 `.db3`）、JSONL、`tasks/team15_hold/INSTALL_RECEIPT.json`，以及箱重、是否空手、是否有人支撑和试验时长的文字记录。

## 配置与仿真的边界

- 配置写在 `tasks/team15_hold/task.py`，继承官方 `K1_CFG`。run Kp/Kd、参考姿态、动作缩放来自已验证的托箱仿真；这些是本 task 的参数，不是声称官方 walking 默认值适合托箱。
- 准备使用官方 K1 的 prepare Kp/Kd，但准备目标换成本模型参考姿态。原版实际电机控制不会自动执行 `effort_limit` 力矩裁剪，也没有独立 bridge 的 watchdog。没有把独立 bridge 的保护伪装成这个 task 的功能。
- 官方 `--mujoco` 只支持机器人本体，没有自由箱子。`python3 scripts/deploy.py --task k1_team15_hold --mujoco` 只能做无箱预览，不能作为带箱测试。
- 带箱测试使用既有自由箱子模拟器加载**同一个新 Policy**，结果在 `validation/`。仿真开始时手臂已抬起、箱子已放好，不涵盖真机准备、加载过程或真实材料差异。
- 模型训练箱重范围为 **0.25–1 kg**。另做了单次 20 秒无箱仿真，基础直立检查通过，躯干最大倾角约 3.21°、最大水平漂移约 1.78 cm。这一次结果不能代表所有空手条件，更不能代替真实机器人测试。

这份插件不需要把机器人接入本地 Windows viewer；控制和 ROS bag 都在机器人现有环境运行。
