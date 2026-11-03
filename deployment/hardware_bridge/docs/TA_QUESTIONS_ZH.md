# 到现场问 TA：无视觉托箱真机对接

我们要运行 **Booster K1 的无视觉托箱策略**：72 维输入（IMU、关节位置/速度、上一动作），22 个关节位置目标，50 Hz。模型不需要相机、箱子状态或重量输入。已有部署仿真结果；**仿真通过不代表真机通过**。

## 先直接问这一句

> 我们有一个 22 关节、50 Hz 的无视觉托箱模型，需要在 booster_deploy 旁边接真机控制。能否提供这台 K1 已经验证过的关节/IMU 映射、控制参数和准备/退出流程？如果有现成的配置文件或控制代码，直接给我们文件就好。我们先只读检查，再和你一起做有支撑的测试。

## 必须带回来的六项信息

可以让 TA 给**现成配置、源代码或文档链接**，不必在现场口述 22 个关节的数值。文件必须对应这台机器人及当前固件，不能只给仿真 XML。

待填写文件是 [site.template.json](../profiles/site.template.json)。`null` 表示尚缺现场信息，不是零或可用默认值。确认项必须对应实际核对结果；`simulation_only` 测试配置不能用于真实电机。

| 问 TA 什么 | 需要留下什么 |
|---|---|
| **1. 在哪里运行，是什么版本？** | K1 型号/硬件版本、机器人标识、固件版本；机器人或控制电脑的 IP、SSH 用户名、联网方式；Linux、Python、ROS 发行版；`booster_interface` 版本；`booster_deploy` commit。不要把密码写入共享文件。 |
| **2. 这 22 个数到底对应什么？** | `motor_state_serial` 每个索引对应的关节名、角度/角速度/力矩单位、正方向、零位偏置；`LowCmd` 的 SERIAL 命令是否使用同一索引及同一坐标；并联关节是否已经由接口换算成关节坐标。 |
| **3. IMU 的方向和单位是什么？** | `rpy` 的顺序、弧度/角度、旋转约定及参考坐标；`gyro` 是机身还是 IMU 坐标、单位；IMU 到策略机身坐标的固定旋转。静止及已知小幅姿态下的读数可用于核对。 |
| **4. 电机应该怎样控制？** | 逐关节的真机 Kp/Kd、允许位置范围、速度上限、力矩上限、目标变化速率；`MotorCmd.weight` 的含义和正确值；串联/并联执行器需要何种参数变换。确认 50 Hz 策略/命令频率是否受支持，是否需要独立更高频的命令刷新。 |
| **5. 谁有控制权，出错后谁停止？** | `/low_state`、`/joint_ctrl`、RPC 的实际名称、消息定义、QoS、ROS_DOMAIN_ID；哪些现有程序也发布控制命令，如何交接保证唯一命令源；传感器或控制进程中断时，固件 watchdog 的真实超时和动作。软件检测不能替代固件断流保护。 |
| **6. 怎样开始、放箱、退出？** | 允许开始的姿态、姿态容差；从该姿态到抬臂参考姿态的已审核关节路点和各段时长；准备阶段的 Kp/Kd；谁/何时放箱或拿箱；防坠/支撑安排；实体急停方法；正常退出、故障退出应进入什么模式以及该模式的实际效果。 |

还请确认可接受的控制时限：状态过期、指令过期、推理耗时、模式切换/查询、命令接收端等待、准备/等放箱/运行的最长时间；以及进入策略前的姿态误差、关节速度和稳定等待时间。这些阈值写入 `limits` 与 `prepare`，不能用网络连通代替实时性核对。

**特别说明：**训练仿真的 Kp/Kd 不能直接当作这台机器的控制参数。固定版本上游代码的 `effort_limit` 配置并未在实际 `ctrl_step()` 中执行软件力矩裁剪；要确认固件/驱动究竟如何限力。`damping` 不是主动站立，`Space` 不是实体急停。

## 关节映射请按这个含义确认

对于每个策略关节，记录其硬件 serial 索引、`sign`（+1 或 -1）和 `offset`（硬件坐标中的弧度值）：

```text
q_model = sign × (q_serial - offset)
dq_model = sign × dq_serial
q_serial_command = offset + sign × q_model_target
```

这一公式只适用于接口已经输出对应关节坐标的情况。若接口暴露并联电机坐标，需要 TA 提供实际变换，不能仅用正负号替代。策略关节名、默认角度和动作尺度在冻结部署包的 `tasks/team15_proprio/models/proprio_hold.contract.json` 中。

## 可让 TA 运行的只读采集命令

在机器人/控制电脑的 **ROS 环境已加载**、当前控制流程保持不变时执行。这些命令不会发布控制目标或切换模式。使用现场 ROS 对应的 Python，不要为了模型升级系统 Python。

```bash
uname -m
python3 --version
printf 'ROS_DISTRO=%s\nROS_DOMAIN_ID=%s\n' "$ROS_DISTRO" "$ROS_DOMAIN_ID"
python3 -c "import rclpy; from booster_interface.msg import LowState, LowCmd, MotorCmd; print('ROS message imports OK')"
ros2 topic type /low_state
ros2 topic info /low_state -v
ros2 topic type /joint_ctrl
ros2 topic info /joint_ctrl -v
ros2 interface show booster_interface/msg/LowState
ros2 interface show booster_interface/msg/LowCmd
ros2 interface show booster_interface/msg/MotorCmd
ros2 service type /booster_rpc_service
ros2 interface show booster_interface/srv/RpcService
```

在现场 `booster_deploy` 仓库目录中再运行：

```bash
git rev-parse HEAD
git status --short
```

把输出保存为文本带回即可。若话题名不同，先由 TA 给出实际名称再替换；不要为了匹配示例去改机器人网络或 ROS 配置。消息类型定义不能证明标定正确，仍需上面的映射/参数文件。

## 现场测试顺序

1. **收集与审核配置。** 对齐消息、关节/IMU 坐标、真机控制参数、watchdog 和准备/停止方案。缺少必填项时停留在只读模式。
2. **只读检查。** 采集真实状态、运行模型，确认维度、单位、方向、数值和推理时延。输出没有发给电机，不能据此宣布真机托箱成功。
3. **TA 主持的支撑下控制检查。** 先验证控制权、目标方向、准备过程和退出/断流行为；使用 TA 提供的姿态和参数。不要用随意填写的模板默认值上机。
4. **托箱闭环测试。** 准备/放箱过程可执行且经现场核对后，再测试策略。以 0.5 kg 为拟议首个仿真对应点，实际可用质量由 TA 根据机器人和支撑条件确认。记录整段状态、目标、故障与人工中止，不只保留成功片段。

模型已有测试从**双臂已抬起、箱子已经由前臂托住**开始；未验证从垂臂抬起、空手运行、抓取、接箱/卸箱或退出后继续站立。上述现场步骤不能把这些未测过程变成默认可靠行为。

## 箱子信息：用于实验记录，不能替代控制配置

- 箱子总质量（包含配重和固定件）、长宽高、配重位置/重心偏移。
- 箱底与前臂接触材料、接触位置；是否使用垫片、绑带或托盘。若新增固定结构，要说明与仿真自由箱体的区别。
- 仿真参考：前后 20 cm × 左右 40 cm × 高 12 cm，质心居中；箱子—前臂摩擦系数 0.7，脚底—地面 1.0，未按实物标定。

英文提问见 [TA_QUESTIONS_EN.md](TA_QUESTIONS_EN.md)。旧仿真包的事实和限制保留在 [现场说明](../../team15_proprio_package/TA_ONSITE_ZH.md) 中；新增控制实现需单独报告实际通过的测试。
