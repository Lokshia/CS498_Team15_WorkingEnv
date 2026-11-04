# Team15: information needed for the real K1 control interface

## Short message to the TA

> We have a no-vision box-holding policy for the K1: 72 inputs from the IMU, joint positions/velocities and previous action, with 22 joint-position actions at 50 Hz. It passed deployment simulation, but we have not validated real motor control. Could you share the joint/IMU mapping, real motor settings, and preparation/exit procedure already tested on this robot? Existing configuration files or control code are preferable to manually entering every value. We would first run a read-only check, then test under your supervision with physical support.

## Required before sending motor commands

The incomplete profile is [site.template.json](../profiles/site.template.json). A `null` is missing site information, not a usable zero/default. Review confirmations must reflect actual checks; `simulation_only` test profiles must not operate real motors.

1. **Robot and runtime:** K1 hardware variant and identifier, firmware version, host architecture, Linux/Python/ROS versions, `booster_interface` version, and the lab's `booster_deploy` commit. Please provide the connection method, IP and SSH username separately from any password.
2. **Joint interface:** The serial index/name mapping for all 22 joints, units, signs and zero offsets. Does SERIAL `LowCmd` use the same coordinates as `motor_state_serial`? For parallel actuators, are these joint coordinates or motor coordinates? If motor coordinates, we need the actual transform.
3. **IMU:** RPY order, units and rotation convention/reference frame; gyro units/frame; and the fixed IMU-to-policy-body rotation. Static readings and readings at a known small pose change help verify the mapping.
4. **Control settings:** Per-joint real Kp/Kd, position/velocity/torque limits, target slew limits, and `MotorCmd.weight` semantics/value. Are 50 Hz commands supported, or is a separate faster command refresh required? Please provide settings valid for the actual parallel-actuation interface, not training-simulator gains.
5. **Authority and failure handling:** Actual state/command/RPC names, message definitions, QoS and ROS_DOMAIN_ID. Which process currently owns motor commands, and how do we guarantee one command source during handover? What does the firmware do when commands stop or the process dies, and after what timeout?
6. **Preparation and stop procedure:** Allowed starting pose/tolerance; reviewed raised-arm waypoints, segment durations and preparation gains; physical support and box-loading/unloading procedure; verified emergency stop; and the actual behavior of normal/fault exit modes. Damping is not active standing.

Please also specify allowable state age, command age, inference time, mode switch/status-query time, command-subscriber wait, and preparation/ready/run durations. The policy-entry check needs a pose-error tolerance, a joint-speed threshold and a stable dwell duration. A successful network connection does not verify these timing requirements.

For the simple sign/offset case, our requested convention is:

```text
q_model = sign * (q_serial - offset)
dq_model = sign * dq_serial
q_serial_command = offset + sign * q_model_target
```

This assumes the interface already exposes joint coordinates. A parallel motor-coordinate transform cannot be replaced by a sign flip. Model joint order, nominal positions and action scales are in `proprio_hold.contract.json` in the frozen deployment package.

The upstream real control path does not apply its `effort_limit` array as a software torque clamp. Please identify the actual firmware/driver protections and watchdog behavior. A software timeout alone cannot protect against a frozen host process or a lost network connection.

## Read-only collection

Using the lab's already loaded ROS environment and its compatible Python, collect:

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

From the lab's deployment repository, also collect `git rev-parse HEAD` and `git status --short`. Adjust topic names only to the actual lab interface. These queries do not publish motor commands or change robot modes; interface definitions do not, by themselves, prove the calibration is correct.

## Proposed on-site stages

- Review the site configuration and failure/exit procedure first. Missing required values keep the controller in read-only mode.
- Run read-only sensing and inference; check units, mapping, valid values and timing. This is not a closed-loop balance test.
- Under the TA's physical support procedure, verify command authority, direction, preparation, interruption and failure handling before testing the learned policy.
- Then perform a recorded holding trial with the prepared arms and supported box. We propose 0.5 kg to match the first simulation case; the TA should determine the allowable physical load and support arrangement.

The existing model tests start with **raised arms and a box already resting on the forearms**. They do not validate arms-down transitions, empty-hand operation, grasping, loading/unloading, or stable standing after exit. There is no camera or box-state input.

Record the physical box's total mass, dimensions, center-of-mass offset and contact materials. The simulation box is 20 cm front-to-back × 40 cm left-to-right × 12 cm high, with a centered COM. Box/forearm friction is 0.7 and sole/ground friction is 1.0; these have not been calibrated to the real materials. Straps or trays change the physical setup and must be documented separately.

See [the Chinese checklist](TA_QUESTIONS_ZH.md) for the student-facing version. The frozen package's simulation results remain distinct from any new hardware test.
