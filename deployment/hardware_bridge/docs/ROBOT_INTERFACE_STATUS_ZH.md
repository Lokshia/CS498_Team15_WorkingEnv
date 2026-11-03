# 现场接口输出：已确认与仍需核对

根据你发来的 `booster@robot` 终端输出，实际通信消息与本适配器使用的接口一致。下面是**用户提供的现场证据**，不是本机直接连接机器人得到的测试。

## 已经确认

| 项目 | 现场结果 |
|---|---|
| 处理器 / Python | `aarch64` / Python 3.10.12 |
| ROS | Humble；`rclpy` 和 Booster 消息可导入 |
| 状态话题 | `/low_state`，类型 `booster_interface/msg/LowState` |
| 状态字段 | IMU 的 `rpy/gyro/acc`，serial/parallel 电机数组；电机含 `q/dq/tau_est` 等 |
| 命令话题 | `/joint_ctrl`，类型 `booster_interface/msg/LowCmd` |
| 命令类型 | `CMD_TYPE_SERIAL=1`；每个命令含 `q/dq/tau/kp/kd/weight/mode` |
| RPC | `/booster_rpc_service`，请求 `api_id/body`，响应 `status/body` |
| 当时的连接数 | `/low_state` 有 1 个发布者；`/joint_ctrl` 有 0 个发布者、2 个订阅者 |

`ROS_DOMAIN_ID` 环境变量为空本身不是报错。不要因此修改已经能发现话题的网络或 domain 配置。此输出也不能单独确定所有进程的实际 DDS domain 设置。

`/joint_ctrl` 的两个订阅者是接收方，不表示有两个程序同时发命令。零发布者只是当时 ROS 图的观测，不证明机器人固件内部没有自己的控制流程。

## 已补上的官方默认值

使用 [k1.upstream-reference.json](../profiles/k1.upstream-reference.json) 查看有来源的预填参考配置；原始全空模板仍保留。来源版本固定为 `booster_deploy` 的 `7bb1462e2ac742eba5c01659b815e2e99fbe617f`。每项依据和三个官方文件的 SHA-256 在 [来源记录](../profiles/k1.upstream-reference.sources.json) 中。

已经填入：

- 与现场一致的话题及 RPC 名称。
- 官方 K1 的 22 关节 serial 顺序。
- 官方运行阶段的逐关节 Kp/Kd。
- 官方控制器使用的 `MotorCmd.weight=0`。
- 官方直接使用 serial 关节值和 IMU 值的约定，即额外符号为 +1、零位修正为 0、额外 IMU 旋转为单位阵。

最后两项映射是**官方程序的约定，不是这台机器人已经测过的标定**。参考配置中的 `review` 仍未勾选，运行增益对托箱策略的适用性也未验证。

官方准备阶段有另一组 Kp/Kd 和准备姿态，已单独放进来源记录，没有混进运行增益。当前 bridge 只支持同一组增益贯穿准备与策略，不能假称已经复现官方分阶段准备流程。官方 `effort_limit` 也不能当作真实 `ctrl_step()` 已经执行力矩裁剪的证据，因此没有直接填成反馈力矩故障阈值。

## 下一步只需要补什么证据

1. **一条实际 `/low_state` 数据**：消息定义里是变长数组，当前输出没有告诉我们实际 `motor_state_serial` 是否恰好 22 个，也没有给出当前姿态/IMU 数值。
2. **现场正在使用的 K1 配置和 commit**：确认它是否与上述官方基准相同，有没有修改关节约定、IMU 或控制参数。
3. **现场准备与退出方案**：固件断流保护、控制权交接、怎样进入抬臂姿态和放箱，以及停止后的实际动作。

新加的软件限速、跟踪误差、倾角和超时等阈值属于适配器设计参数，不是要求 TA 从机器人里查出一组“原厂值”。需要结合真实状态和现场约束确定并测试。

这些输出确认了**接口结构**。它们还没有验证真实电机运动、参数标定或托箱平衡。参考配置保留未确认项时，预检仍会拒绝 live 运行；不要为了消除报错任意填数或把审核项全部改为 `true`。
