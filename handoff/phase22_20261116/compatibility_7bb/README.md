# Site version addendum: 7bb1462

This optional reference applies only to upstream commit `7bb1462e2ac742eba5c01659b815e2e99fbe617f`, observed at the site. It accompanies the unchanged phase22 ZIP; it does not replace the original cecc reference or weaken its checks.

- Seven of the original eight pinned core files are identical to cecc, including the K1 mapping, robot controller, base controller, configuration, registry, math, and ONNX runner.
- The older `scripts/deploy.py` lacks the optional `--checkpoint` override. This task uses its own configured ONNX and does not need that override.
- The additional ninth checked file is `remote_control_service.py`. 7bb lacks a later generic-HID face-button mapping change. Verify physical controller button layout at the site; do not assume the later mapping applies.
- The supplied nine expected hashes come from the named Git commit's objects, not from arbitrary files currently installed on a robot.

[Reviewed reference](7bb_core_reference.json) · [Source differences](cecc_to_7bb_core.diff) · [Local offline audit](AUDIT.json)

Reference file SHA256: `b0ae518c924e7b9d2492f5cb3808db60aa67b1b654740f417ef7595f3ea4de06`.

## Results and scope

The unchanged phase22 task was installed into a clean local 7bb checkout. Task registration, all 516 model vectors, and the original CPU runner passed. Maximum numerical error was `3.7550926208496094e-06`. The original preflight, explicitly using this reference, exited 0. All nine checked upstream files remained unchanged. A separate run using the original cecc reference still rejected 7bb as expected.

These are local Windows offline checks, not ARM64/ROS timing, a new stability test, or real robot validation. The robot's existing untracked course files and `scripts/cs498_deploy.py` are not included in this clean-checkout audit. This reference checks `scripts/deploy.py`.

## On the robot

Load the existing Booster ROS environment. Save the reviewed JSON outside the extracted archive, and verify its hash. From the extracted phase22 package root:

```bash
python3 scripts/ta_readonly_preflight.py \
  --deploy-root ~/booster_deploy \
  --reference ~/cs498rt/team15/7bb_core_reference.json \
  --benchmark --native-runner --ros-seconds 3 \
  --output ~/cs498rt/team15/preflight_7bb_trial01.json
```

Use a new output filename for each run. This reads robot state and checks the model and nine source files. It does not construct the motor controller, publish motor commands, or call mode-switch RPCs. Review all results and the native runner timing on the actual robot. The report does not grant control ownership or prove hardware stability. Keep `hardware_validated: false`.

中文：这是针对现场原版7bb版本的独立核对记录。无需切换机器人仓库版本，也无需替换已经校验过的ZIP。原cecc检查继续保留；这里只通过显式`--reference`选择已经审查过的7bb源码基准。真机数据接收、性能、支撑、停止流程和控制权仍需现场确认。
