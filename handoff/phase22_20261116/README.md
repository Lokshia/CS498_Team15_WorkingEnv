# Team15 K1 carry deployment handoff — phase22 / 2026-11-16

**TA: download the complete ZIP, extract it, then read `START_HERE_ZH.md` and `README_ZH.md` inside.** The archive contains the runnable task, model, checks, recording tools, and matching simulation evidence. No training environment is needed on the robot.

[Download the complete deployment ZIP (65.7 MB)](Team15_K1_Carry_phase22_TA_20261116.zip?raw=true) · [SHA256](SHA256SUMS.txt) · [Delivery receipt](DELIVERY_RECEIPT.json) · [Installation audit](INSTALL_AUDIT.json) · [Detailed Chinese/English handoff](TA_SCOPE_ZH.md)

## Integration

**Site at upstream `7bb1462`?** Read the [separate reviewed 7bb compatibility addendum](compatibility_7bb/README.md) and use its explicit read-only reference. The original ZIP and cecc qualification remain unchanged.

- Robot: Booster K1. Intended site environment: aarch64, Python 3.10, ROS Humble.
- Register **`k1_team15_carry`** in the existing original `booster_deploy`; use its `scripts/deploy.py` and existing controller.
- Reference upstream commit: `cecc00fc019de3c640af2d0421059b59a2e080c1`.
- Actor inputs: proprioception and velocity commands. No camera, motion capture, box tracking, or box-mass input.
- Keep the task source, `policy.onnx`, `policy.contract.json`, and `policy.test_vectors.npz` together. Do not replace the walking model alone.
- This PR publishes a standalone deployment artifact. The older training code elsewhere in this repository is not the source needed to install this task.

## Offline checks first

From the extracted package root, install into the TA's already working deployment checkout:

```bash
python3 install.py --deploy-root ~/booster_deploy
cd ~/booster_deploy
python3 scripts/deploy.py --list
python3 -m tasks.team15_carry.check
```

If a different `tasks/team15_carry` already exists, the installer refuses to overwrite it. Stop the old task and back it up outside the upstream `tasks` directory before installing this version.

From the extracted package root, run the read-only site check:

```bash
python3 scripts/ta_readonly_preflight.py \
  --deploy-root ~/booster_deploy --benchmark --native-runner \
  --ros-seconds 3 --output preflight_trial01.json
```

These checks do not command motors. Actual deployment and ROS bag commands are in the archive's `README_ZH.md` and `docs/RECORDING_ZH.md`. Use that complete procedure with the TA's existing support, control-ownership, and stop procedure. `X/x` preparation already moves the robot; space only clears velocity requests. Empty-hand qualification is zero-command standing only.

## What was checked

| Check | Result and scope |
| --- | --- |
| Native deployment simulation | 15/15, five scenarios × three benchmark seeds, 20 seconds each; `walking_recovery_window_v3` |
| Empty hands | 3/3 zero-command standing trials; no empty-hand walking qualification |
| Additional GPU combinations | 8/8, reported separately |
| Additional previously unused seed 91827 | 3/3 stand/walk/combined trials, reported separately |
| Late disturbances | Separate 35-second diagnostic, push at 22 seconds and downward pressure at 28 seconds |
| Clean upstream install | First install, repeat install, task registration, offline check, and native CPU runner passed locally on Windows |
| Model parity | 516 vectors, maximum absolute error `3.7551e-6` |
| Upstream core | Eight pinned official core files unchanged |

The declared loaded simulation condition is a **0.5 kg box and 0.1 m/s forward motion**, with specified 8 N / 0.3 s horizontal pushes and 3 N / 1 s downward pressure. These are simulation conditions, not instructions for applying force to a real robot or rated payload limits. Box dimensions: 0.20 × 0.40 × 0.12 m; free body, no weld, light chest contact allowed with both arms supporting it. Friction values are not physically calibrated.

**This updated model is simulation-qualified, not hardware-validated.** About 5 degrees of residual lean remains; a late pressure test still shows unequal foot loading, and high-frequency leg motion after a push remains above the old baseline. Arm raising, box placement, control-mode transitions, 3–5 kg payloads, and arbitrary disturbances are not validated. Local CPU timing does not establish robot ARM64/ROS timing.

## 给 TA 的简短说明

这一包是最新 phase22 无视觉托箱任务，沿用原版 `booster_deploy` 的控制器，以 `k1_team15_carry` 注册。请下载整个 ZIP；不要只拿 ONNX，也不需要本项目的 Windows 虚拟环境或训练权重。包内包含安装器、任务源码、配套模型/接口、离线与只读检查、ROS bag 脚本及同模型仿真结果。

仿真范围是 0.5 kg 和慢速 0.1 m/s；最新权重尚未做真机验证。先按包内步骤安装与检查，再由 TA 按现场已确认的支撑和停止流程进行零速度测试、放箱和慢速行走。

## Integrity

```text
ZIP SHA256:
73325a8370c4df0b248e4a8800c20b0e33900125b93afdc861e28b71754a0add
ONNX SHA256:
66a259aa480bd6eb6f4fc9667ebdb3bbf1464b85f805e4cf23edcb7623edadb4
```

On Linux, run `sha256sum -c SHA256SUMS.txt` beside the downloaded ZIP. The archive also includes its per-file manifest.
