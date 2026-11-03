# 无视觉托箱：真机控制适配候选

这个目录增加了**实际 ROS 2 电机发布、模式切换和中止路径**。默认只做本地预检；现场配置未填完整时拒绝运行。原来已验证的 `team15_proprio_package/` 和 ZIP 保持原样。

**状态：代码和模拟接口可测试，尚未连接真实机器人。** 软件测试及部署仿真不能证明真实机器人会稳定托箱。仍需 TA 提供这台机器的标定、参数和准备/停止方案，再做现场验证。

- [现场问 TA 的中文清单](docs/TA_QUESTIONS_ZH.md)
- [可以直接发给 TA 的英文清单](docs/TA_QUESTIONS_EN.md)
- [必填现场参数模板](profiles/site.template.json)
- [模型接口与原始仿真结果](../team15_proprio_package/README_ZH.md)

## 控制流程

```text
预检（无 ROS）
  → WAITING：只订阅状态，等待 arm 机器人标识
  → 核对起始姿态、命令订阅者和唯一发布源
  → ARMING：预先发送当前姿态保持目标，再请求并确认 Custom
  → PREPARING：按 TA 审核的路点平滑移动，逐步限速，可中止
  → READY：保持参考姿态；确认实际关节已到位、低速并稳定一段时间
  → 操作员按现场放箱/支撑方案准备后，输入 run
  → RUNNING：50 Hz 无视觉策略，限制目标并反馈实际已发布的上一动作
  → STOPPED：停止后不自动重启；尝试并核实 TA 指定退出模式
```

`arm` 后就可能产生动作，不是只读命令。`run` 只允许在 READY 后使用。本程序不会检测箱子是否已放好，不会自动抓箱，也不会因为没有视觉而知道箱子已经掉落。

ROS 接口沿用固定版本 `booster_deploy` 的 `/low_state`、SERIAL `LowCmd`、模式 RPC；使用独立控制入口，不依赖上游不可中断的准备循环。没有替换上游 walking 策略。现场消息/坐标若不符合模板支持的协议，应修改适配并重新测试，不能把字段强行填写成看似匹配。

## 先向 TA 拿什么

最好拿这台 K1 已经用过的**配置文件、接口定义和启动/退出代码**。核心是 22 关节的索引/方向/零位、IMU 旋转、真机 Kp/Kd 与各项限制、独立固件断流保护，以及可执行的抬臂/放箱/停止方案。详细问题和只读采集命令在上面中英文清单中。

模板中 `null` 代表未知，必须填真实值；复选确认项也必须对应实际审核。填写完成只代表程序能检查输入格式，不证明标定或方案正确。`tests/fixtures.py` 和评测中的 `simulation_only_profile.json` 是软件测试参数，**不能用于真实电机**，`--live` 会拒绝它们。

## 安装与预检

Mac 可以下载并通过 SSH 操作机器人。真实 ROS 程序运行在机器人/实验室控制机的 Linux 环境中。使用 TA 的 ROS 对应 Python，不要复制 Windows `.venv`，也不要升级实验室系统 Python。

在本仓库根目录，用 TA 同意的独立环境安装推理依赖。例如现场已加载正确 ROS 环境后：

```bash
python3 -m venv --system-site-packages .venv-team15
source .venv-team15/bin/activate
python -m pip install -r deployment/hardware_bridge/requirements.txt
python -c "import rclpy, numpy, onnxruntime; from booster_interface.msg import LowState, LowCmd, MotorCmd; print('imports OK')"
```

复制模板为本地 `site.local.json`，与 TA 一起填写。此类现场配置和日志默认不上传 Git（包含设备标识与现场细节）。模型已在相邻冻结包里，不需要另外找权重。

```bash
cp deployment/hardware_bridge/profiles/site.template.json site.local.json
python deployment/hardware_bridge/run.py --config site.local.json --preflight
```

预检检查完整参数、旋转矩阵、路点/速度、模型与合同 SHA，以及全部 534 条推理测试向量。不启动 ROS、不创建发布器、不切模式。缺参数会一次列出需要补全的字段。

## 只读检查

配置完整且现场 ROS 环境已加载时：

```bash
python deployment/hardware_bridge/run.py --config site.local.json --shadow --seconds 20 --output logs/team15_shadow_01
```

只订阅并运行推理，**不创建电机发布器或模式客户端**。每帧上一动作保持零，属于冷启动诊断，不是真机闭环回放。输出目录必须是新目录。

## TA 主持的实际控制测试

完成真实标定、独占控制权、固件 deadman、实体急停/支撑和进入/退出方案核对后，在**机器人本机**运行：

```bash
python deployment/hardware_bridge/run.py --config site.local.json --live --seconds 20 --output logs/team15_live_01
```

启动后先只读等待。操作员依次输入：

```text
arm <site.local.json 中 review.robot_serial 的值>
```

这会发送保持指令、进入 Custom，并执行审核过的准备路点。到 READY 后仍保持姿态，按 TA 的实际支撑/放箱安排操作，再输入：

```text
run
```

策略最长运行 `--seconds` 和配置 `run_timeout_s` 中较短的时长。随时输入 `stop` 或 Ctrl+C 会触发软件中止；SSH 终端正常关闭也会中止。网络断开未必及时关闭进程，因此不把 SSH、键盘或 Python 当实体急停。

停止会锁住后续发布，并在独立线程尝试指定退出模式；失败明确记录 `EXIT_UNCONFIRMED`。若检测到机器人已被外部控制切出 Custom，保留外部模式，不重新抢占或切回 walking。重新开始必须结束进程并重新做现场检查，不支持自动重试。

## 具体检查与局限

- `q_model = sign × (q_serial − offset)`，命令使用其逆变换；IMU 重力和 gyro 旋转到策略机身坐标。
- 位置目标受现场限制和原验证场景的策略范围双重约束，再执行每步变化限制；模型输入使用成功**本地发布**目标对应的上一动作。
- 持续检查位置、速度、反馈力矩、躯干倾角、接收数据年龄、推理时延、命令更新时限和 Custom 模式。并行 watchdog 在推理阻塞时也能锁住发布并尝试退出。
- 反馈力矩检查是故障检测，不是直接力矩限幅；真实限力必须由驱动/固件提供。Kp/Kd 的电机侧含义、并联执行器变换不能从仿真推断。
- ROS 图上的唯一发布者/订阅者检查不等于设备确认了指令；`publish()` 返回也不是电机 ACK。接收时间新鲜不代表传感器原始时间戳正确。固件失联保护必须独立确认。
- Python/ROS 不是硬实时系统。RPC 超时或取消不保证机器人没有收到旧请求；若退出未确认，执行 TA 的实体停止/支撑方案。
- 当前使用同一组经审核的 Kp/Kd 贯穿准备和策略；若现场需要分阶段增益或更高频命令刷新，需要扩展后再测。
- 只有 `q/dq/tau` 已是对应关节坐标的 SERIAL 接口受支持；电机坐标、额外关节或不同 IMU 约定需要额外适配。

日志保存配置、真实输入、候选动作、发布目标、阶段与故障。`commands_published` 是本地发布次数，不能称为硬件成功次数。完整实际运动、准备/加载、物理急停行为及真机托箱效果尚未验证。

## 软件与仿真复测

不连接机器人即可运行：

```bash
python -m unittest discover -s deployment/hardware_bridge/tests -v
```

需要额外 MuJoCo/torch 和安装好冻结仿真任务的 pinned 仓库，才能运行新的动作处理链闭环验证：

```bash
python deployment/hardware_bridge/tests/run_native_bridge.py --deploy-root /path/to/booster_deploy --assets-root /path/to/booster_assets --output logs/native_bridge_01 --episodes 3 --seconds 20 --mass-kg 0.5
```

该测试使用模拟器参数，双臂已抬起、箱子已放好；不测试 ROS、物理准备或真实电机。新增限速改变了策略闭环，因此结果单独记录，不能直接套用旧包的 21/21。最新记录见本目录 `validation/`（含完整轨迹）。
