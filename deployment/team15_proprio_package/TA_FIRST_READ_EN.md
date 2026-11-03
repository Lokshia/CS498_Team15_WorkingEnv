# Team15 — proprioceptive box holding

**Current status: deployment simulation passed after correcting the box's local center of mass.** The unchanged policy passed six 20-second trials at each of 0.25, 0.50 and 1.0 kg, plus three 60-second trials at 0.50 kg. No retraining or changes to rewards, friction or pass thresholds were used. Use **0.50 kg** for the first demonstration. The 1.0 kg trials reached 9.624° box tilt, close to the 10° limit.

This package contains a **simulation task** for a pinned `booster_deploy` checkout, an actor-only ONNX/TorchScript model, interface/test vectors, validation tools, and a **read-only ROS 2 diagnostic**. It does **not** contain an enabled hardware-actuation task. Watch `demo/demo_0p5kg.mp4`; full results are in `reports/deploy_0p25kg/`, `reports/deploy_0p5kg/`, `reports/deploy_1kg/` and `reports/deploy_0p5kg_60s/` (each contains `report.json`).

The actor runs at 50 Hz: body gyro (3), body-frame unit gravity (3), joint position minus reference (22), joint velocity (22), previous action (22). No camera, motion capture, box state, mass input or slip sensor is needed. Normalization and action clipping are inside the model. The adapter applies joint offsets/scales and XML target limits. This is not a drop-in replacement for the default walking policy.

The box is a free rigid body supported by the forearms: 0.20 × 0.40 × 0.12 m; centered uniform inertia. Simulated box–forearm sliding friction is 0.7 and sole–ground friction is 1.0, neither calibrated to the real materials. The old deployment failures used an incorrect local inertial offset outside the box; those results do not describe the corrected scene. The earlier training-simulator 1.0 kg failures are a separate result from a different scene.

After installing dependencies and preparing the pinned sibling checkouts described in `README_ZH.md`, run from their parent directory:

```bash
python team15_proprio_package/scripts/install_task.py --deploy-root booster_deploy
python team15_proprio_package/scripts/validate_package.py --deploy-root booster_deploy --assets-root booster_assets --output sim_check_050kg --episodes 6 --mass-kg 0.5 --seconds 20 --seed 21031
```

Expect exit code 0 and `all_passed: true`. If the laboratory replay fails, keep its report and resolve the difference before adding motor control. Use a fresh checkout if an older Team15 task is installed; conflicting files are not overwritten.

The independent simulation entry, from inside `booster_deploy/`, is:

```bash
python -m tasks.team15_proprio.simulate --mass-kg 0.5 --paused
```

Space pauses/resumes this simulation. Initial conditions already have raised arms and a supported box; arm raising, box loading/unloading and empty-hand operation are not validated by these tests.

For the read-only robot check, return to the parent `test_workspace/` directory and use the laboratory's ROS environment:

```bash
python team15_proprio_package/scripts/shadow_check.py --seconds 20 --output shadow_check_01
```

Use the laboratory's ROS-compatible Python. The script subscribes to `/low_state` (`booster_interface/msg/LowState`) and sends no motor commands or mode RPCs. It saves the sensor fields, model input/output and inference timing. Previous action is reset to zero per frame, so this is a cold-start diagnostic, not closed-loop hardware validation.

Before implementing motor control, we need the actual joint order/zero/sign calibration, IMU transform, motor-side Kp/Kd, effective firmware limits, sensor/command watchdog, and an interruptible raised-arm entry/box-loading/exit procedure. Upstream `effort_limit` is not applied as a software torque clamp in the real `ctrl_step`. Its standing preparation moves to the configured pose in approximately one second; Ctrl+C is not a guaranteed immediate stop. Please use the laboratory's verified emergency-stop and support procedure.

Installation commands and the staged on-site checklist are in `README_ZH.md` and `TA_ONSITE_ZH.md`. The pinned upstream revisions are listed in `PROVENANCE.json`; install in a separate checkout.
