# 真机测试录制（ROS 2 Humble）

录制脚本是观察工具。控制仍由原版 `booster_deploy/scripts/deploy.py --task ...` 入口运行。录制脚本不启动策略、不切换机器人模式，也不发送电机指令。

## 在另一个终端先开始录制

使用与控制程序相同的 ROS 环境、网络和 `ROS_DOMAIN_ID`。先按实验室原来的方式加载 Humble 和 `booster_interface` 工作空间，再进入解压后的本 task 包目录（仓库中是 `deployment/booster_deploy_task/`）：

```bash
python3 -c "import rclpy; from booster_interface.msg import LowState, LowCmd; print('ROS OK')"
bash scripts/record_rosbag.sh
```

脚本每次自动创建新的 `recordings/team15_<UTC时间>_<进程号>_<随机数>/`。也可以给一个尚不存在的目录：

```bash
bash scripts/record_rosbag.sh recordings/empty_hands_trial01
```

已有目录会被拒绝，不会覆盖。录制启动后再进行现场已确认的控制测试。测试结束，回到录制终端按 **Ctrl+C**，等命令退出、`metadata.yaml` 写完后，再复制整个 bag 目录。

```bash
ros2 bag info recordings/empty_hands_trial01
```

确认总时长以及各 topic 的消息数量。某个 topic 没有数据，不等于机器人没有动作：先确认该 topic 名称和发布者。脚本保留正常的 topic discovery；例如启动时 `/joint_ctrl` 没有发布者，随后控制程序启动时才会出现。

## 会录到什么

| Topic | 内容及限制 |
|---|---|
| `/low_state` | IMU 的 rpy、gyro、acc，以及串联/并联关节的 q、dq、ddq、tau_est 等原始状态。`tau_est` 是估计关节力矩，不能直接当作足底接触力。 |
| `/joint_ctrl` | 发布到 ROS 的 LowCmd，包括目标 q/dq、tau、kp/kd 等。是发送的指令，不代表电机实际完全执行。 |
| `/remote_controller_state` | 发布者实际提供的遥控器状态；如果现场未发布此 topic，bag 中可能没有它。 |
| `/rosout` | ROS 节点输出的日志；普通 Python print/stdout 不一定在里面。 |

这些 topic **不包含策略内部的 observation、网络原始 action、滤波/限幅的中间量**。这些数据需要 task policy 自己记录 JSONL，并与 bag 一起保存；bag 录制脚本不会自动生成 policy JSONL。保存时也附上模型、配置文件和试验条件（空手/箱子重量、试验编号）。

这里没有额外的足底力传感器数据、箱子位置或视觉数据。不能仅凭 bag 推断真实箱子滑移或左右脚的准确受力。LowState 没有传感器源时间戳字段；bag 时间通常是录制接收时间，不能据此声称精确的传感器延迟。

## QoS 与离线分析

显式 QoS 使用 `best_effort`、`volatile`。这与已观察到的可靠状态发布者兼容，也能接受 best-effort 发布者，避免录制端请求不支持的可靠传输。它只收录订阅后的实时消息，不能保证不丢包。采样完整性要结合消息数量、时间间隔和日志判断。

**不要在连接真机的 ROS 网络中直接 `ros2 bag play` 这个包。** 包含 `/joint_ctrl`，回放会重新发布电机指令。分析优先读取 bag 文件；需要回放时，先与机器人网络隔离，并由 TA 确认只选择观察 topic，不能回放控制 topic。

## 已核对的 Humble 官方接口

- [ROS 2 Humble rosbag record 参数定义](https://github.com/ros2/rosbag2/blob/humble/ros2bag/ros2bag/verb/record.py)：支持 topic 列表、`--output`、`--storage` 和 `--qos-profile-overrides-path`。
- [ROS 2 Humble QoS 覆盖说明](https://github.com/ros2/ros2_documentation/blob/humble/source/How-To-Guides/Overriding-QoS-Policies-For-Recording-And-Playback.rst)：按 topic 配置 history/depth/reliability/durability。

本地验证使用假的 `ros2` 命令检查参数和不覆盖行为，没有连接机器人；现场是否收到数据，要以 `ros2 bag info` 的实际输出为准。
