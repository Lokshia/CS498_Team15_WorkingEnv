# Team15 无视觉托箱：明天现场怎么测

**带箱部署仿真已通过：0.25 / 0.5 / 1.0 kg各6次20秒，0.5 kg另3次60秒。默认演示用0.5 kg。** 此前失败来自箱子局部质心设在箱体外，现已修正；模型、奖励、摩擦和评分阈值未改，也没有重训。[详细结果](VALIDATION_STATUS_ZH.md)。

**这个包可以安装到指定版本 `booster_deploy` 做带箱仿真，也可以在机器人上读取真实传感器、运行模型并保存候选输出。当前不发送电机命令，尚未提供可接管真机的控制任务。**

新版 actor 只用 IMU、关节角度/速度和上一动作，共72维；不需要相机、箱体状态、重量输入或滑移传感器。它已经解决旧97维模型缺箱体观测的问题，但没有因此完成电机标定、增益和启动过渡的验证。

## 1. 带什么，放在哪里

把完整 ZIP 发给 TA，解压得到 `team15_proprio_package/`；不要只发 ONNX。建议在一个新的测试目录并排放：

```text
test_workspace/
  team15_proprio_package/
    tasks/team15_proprio/models/proprio_hold.onnx
    tasks/team15_proprio/models/proprio_hold.contract.json
    scripts/install_task.py
    scripts/validate_package.py
    scripts/shadow_check.py
    TA_ONSITE_ZH.md
  booster_deploy/
  booster_assets/
```

模型合同包含观测顺序、22关节对应关系、默认姿态、动作缩放和哈希。当前来源checkpoint的SHA256必须是：

`b0722baf7d79d513d54cb8afec74860ac4aed5296bbd8049d73ec9e4afcc1382`

带实际箱子，记录**总质量、尺寸、配重位置和前臂接触材料**。模拟箱子尺寸为20 × 40 × 12 cm、质心居中；先按0.5 kg准备演示。1.0 kg虽通过此次部署仿真，最大箱体倾角9.624°，距10°阈值余量较小。尺寸/质量相似不等于摩擦、重心也相同。本模型从手臂已抬起、箱子已放好开始；还未验证搬放箱子、从垂臂抬起或空手运行。

可以先给TA看 `demo/demo_0p5kg.mp4`。完整结果分别在 `reports/deploy_0p25kg/report.json`、`reports/deploy_0p5kg/report.json`、`reports/deploy_1kg/report.json` 和 `reports/deploy_0p5kg_60s/report.json`。

## 2. 先在副本里安装并运行官方仿真

从 `test_workspace/` 执行。使用包内README说明的仿真Python环境，先核对版本：

```bash
git -C booster_deploy rev-parse HEAD
git -C booster_assets rev-parse HEAD
```

对应已固定的上游版本：

- `booster_deploy`: `7bb1462e2ac742eba5c01659b815e2e99fbe617f`
- `booster_assets`: `12a97516f57469078707afb587a5f84c1cb110d1`

版本不同，应另建固定版本的副本；不要把现场正在工作的仓库强制切换。安装脚本会拒绝版本或文件冲突。

```bash
python team15_proprio_package/scripts/install_task.py --deploy-root booster_deploy
python team15_proprio_package/scripts/validate_package.py --deploy-root booster_deploy --assets-root booster_assets --output sim_check_050kg --episodes 6 --mass-kg 0.5 --seconds 20 --seed 21031
```

输出目录必须是新目录。查看生成的 `report.json`，按实际成功/失败记录；不能因为模型加载成功就算托箱通过。

本包对应本地结果为返回码0、`all_passed: true`。若现场重跑不通过，保留输出目录和错误信息，先排查版本/场景/结果差异，不推进电机控制。通过后可继续做只读检查，但这还不表示已完成真机接管。

若要单独确认箱子惯性定义，可运行已提供的回归检查：

```bash
python team15_proprio_package/tests/test_payload_inertia.py --deploy-root booster_deploy --assets-root booster_assets -v
```

它检查质心、惯量与场景不变量，不替代20秒闭环验证。装过旧版时请使用新工作副本，安装脚本会拒绝直接覆盖旧任务。

打开带箱仿真窗口：

```bash
cd booster_deploy
python scripts/deploy.py --list
python -m tasks.team15_proprio.simulate --mass-kg 0.5 --paused
```

在这个仿真窗口中，Space切换暂停/继续。任务名是 `team15_proprio_hold_sim`。带箱场景使用上面的专用模拟入口；不要把删除 `--mujoco` 理解为已经完成真机适配。

## 3. 机器人上先读数据，不动电机

以下命令在 TA 已配置的机器人/控制电脑上执行。使用**能导入现场 `rclpy` 和 `booster_interface` 的 Python**；不要照搬本机Windows Python 3.13环境。只读推理另外需要兼容该Python的 `numpy`、`onnxruntime`，不依赖训练用PyTorch。

先加载现场ROS环境并检查接口：

```bash
source /opt/booster/BoosterRos2Interface/install/setup.bash
python -c "import rclpy, numpy, onnxruntime; from booster_interface.msg import LowState; print('imports OK')"
ros2 topic type /low_state
ros2 interface show booster_interface/msg/LowState
ros2 topic info /low_state -v
```

本包按固定版本官方源码读取：`imu_state.rpy`、`imu_state.gyro`，以及 `motor_state_serial` 中22个电机的 `q`、`dq`、`tau_est`。消息类型应为 `booster_interface/msg/LowState`。订阅使用 BEST_EFFORT、KEEP_LAST、depth=1。

在机器人由 TA 的原有控制流程保持安全状态时，回到 `test_workspace/` 运行：

```bash
python team15_proprio_package/scripts/shadow_check.py --seconds 20 --output shadow_check_01
```

这一步叫**只读推理检查**：不创建电机发布器，不调用模式切换RPC，不实例化 `BoosterRobotPortal`。不需要按X/A；不会接管机器人。

产生两个文件：

- `shadow_check_01/summary.json`：是否收到了有效数据、推理耗时、错误原因、`commands_sent: 0`。
- `shadow_check_01/samples.jsonl`：真实IMU/关节状态、构造的72维输入、模型候选动作和未投影的候选位置目标。

**先检查这几点：** `source` 为 `live_ros2`；`passed` 和 `completed_requested_duration` 为true；确有采样；22关节/3维IMU数量正确、数值有限；`commands_sent` 为0。`passed` 只说明读数和推理流程完成，不能证明关节方向/零位正确或真机能站稳。时间检查用本机收到消息的单调时钟，不代表已验证传感器时间戳或硬实时性能。

这里每帧把上一动作设为零，属于冷启动输入诊断，**不是闭环策略回放**。候选目标没有发送，也不能从日志复制到电机发布器。数据中断或格式不符会报错停止；停止这个只读程序用 `Ctrl+C`。

想带回电脑排查，可复制整个 `shadow_check_01/`。离线复跑命令：

```bash
python team15_proprio_package/scripts/shadow_check.py --replay-jsonl shadow_check_01/samples.jsonl --output shadow_replay_01
```

离线复跑不需要ROS，也不算新增真机测试。

## 4. 真正让机器人接管前，还需要哪些具体信息

请 TA 留下下面的信息。它们是要写进实际硬件配置/控制实现的数值和流程，不是一个 `allow_hardware=true` 开关。

| 项目 | 需要现场提供/确认的内容 |
|---|---|
| 设备身份 | K1版本、固件版本、ROS版本、`booster_interface`版本、现场 `booster_deploy` commit |
| 22关节映射 | 每个serial index对应的关节名、正负方向、零位偏置、实际位置/速度/力矩限制 |
| IMU映射 | RPY单位/顺序、gyro坐标系、IMU到机身的固定旋转、静止和小幅已知姿态下的读数 |
| 控制参数 | 每个真实电机采用的Kp/Kd、50 Hz策略周期能否满足、目标速度/每步变化限制 |
| 状态/指令失效 | 传感器停止更新、推理超时、程序退出时，哪层watchdog在多长时间后采取什么动作 |
| 准备与交接 | 从当前姿态进入抬臂参考姿态的路径、时长、开始策略条件、箱子何时放上/拿走 |
| 停止流程 | 现场急停方式、防坠措施、退出后应进入的模式及其在该固件上的实际行为 |

官方把并联执行器的Kd直接发到电机；不能把训练仿真的Kd原样当成真机正确值。官方 `ctrl_step()` 发送q/kp/kd，未用配置里的 `effort_limit` 列表做软件力矩裁剪；实际限力/限位保护必须核对固件实现。

上述信息补齐后，仍需实现并测试硬件控制路径，包括真实上一发送动作、目标限制、数据过期检查和可中止的准备过程。**当前交付在“官方仿真＋真实传感器只读推理”处停止，尚不能执行自定义策略接管真机。**

## 5. 官方按键是什么意思

以下仅解释固定版本官方源码，供 TA 审查将来的硬件流程；当前只读检查不使用这些按钮。

| 操作 | 官方代码行为 |
|---|---|
| 遥控器X / 键盘x | 进入Custom准备流程 |
| `prepare_mode=standing` | X后约500×2 ms，即1秒插值到 `prepare_state.joint_pos`；这会使机器人运动 |
| `prepare_mode=walking` | X后运行零速度walking准备，再等待任务切换 |
| 遥控器A / 键盘r | 开始或切换到所选任务策略 |
| Space | 仅清零速度命令；**不是急停**，也不会停止托箱策略输出 |
| Ctrl+C | 请求退出，随后尝试切到配置的退出模式；默认walking，可配置damping |

该版本standing插值循环内没有检查退出事件，模式RPC还可能重试，因此不能把 `Ctrl+C` 承诺为立即停机。damping也不等于主动保持站立。现场应使用 TA 指定的实体急停/防坠办法，并验证具体退出行为。

## 6. 这次能得出什么结论

仿真通过：证明模型在这份仿真配置下工作。

只读推理通过：证明现场消息能够构造输入并运行这个模型；机器人没有执行输出。

真机闭环通过：需要另外完成上述硬件对接和现场测试，当前尚未发生。不要把前两种结果写成第三种。

来源：固定版本官方 `booster_robot_controller.py`、`controller_cfg.py`、`robots/k1.py`、`remote_control_service.py` 和 README；[官方源码](https://github.com/BoosterRobotics/booster_deploy/tree/7bb1462e2ac742eba5c01659b815e2e99fbe617f)。
