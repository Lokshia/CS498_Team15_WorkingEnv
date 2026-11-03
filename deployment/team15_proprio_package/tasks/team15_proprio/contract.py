"""Training-side constants, not approved hardware motor settings.

The lists use the actor's natural joint order, not Isaac's interleaved order.
Names are explicitly translated to booster_deploy's K1 asset names. Units are
radians, rad/s, seconds and N*m. Do not reuse these simulation gains on motors
without reviewing the real parallel-actuation interface and limits.
"""
from __future__ import annotations

import math
import sys
import argparse
from collections.abc import Sequence

TASK_NAME = "team15_proprio_hold_sim"
UPSTREAM_COMMIT = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"
HARDWARE_READY = False
CONTROL_DT = 0.02
OBSERVATION_DIM = 72
ACTION_DIM = 22

TRAINING_JOINT_NAMES = (
    "Head_Yaw", "Head_Pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
JOINT_NAMES = (
    "aahead_yaw_joint", "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
DEFAULT_JOINT_POS = (
    0., 0.,
    0., -math.pi / 2, 0., -(math.pi / 2 - .06),
    0., math.pi / 2, 0., math.pi / 2 - .06,
    -.4, 0., 0., .8, -.4, 0.,
    -.4, 0., 0., .8, -.4, 0.,
)
ACTION_SCALE = (
    .375, .375,
    .35, .35, .35, .35, .35, .35, .35, .35,
    .2125, .2375, .1196875, .35, .1915, .1915,
    .2125, .2375, .1196875, .35, .1915, .1915,
)
SIM_STIFFNESS = (4., 4.) + (10.,) * 8 + (80., 80., 80., 80., 50., 50.) * 2
SIM_DAMPING = (.25, .25) + (1.,) * 8 + (4., 4., 4., 4., 2., 2.) * 2
SIM_EFFORT_LIMIT = (6., 6.) + (14.,) * 8 + (68., 76., 38.3, 112., 38.3, 38.3) * 2
# Official booster_assets 12a9751 K1 geometry is not the training asset. With
# training q_default, root z=.5125 penetrates both foot boxes by .005573700934m.
# Lift by that clearance plus 0.1 mm; no force or pose constraint is added.
SIM_INIT_POS = (0., 0., .5181737009342727)
SOURCE_CHECKPOINT_SHA256 = "b0722baf7d79d513d54cb8afec74860ac4aed5296bbd8049d73ec9e4afcc1382"

OBSERVATION_SLICES = {
    "base_angular_velocity_body_rad_s": (0, 3),
    "unit_gravity_body": (3, 6),
    "joint_position_minus_default_rad": (6, 28),
    "joint_velocity_rad_s": (28, 50),
    "previous_clipped_action_unscaled": (50, 72),
}

REAL_ROBOT_BLOCK_REASON = (
    "Team15 task is SIMULATION ONLY. No reviewed hardware adapter is included: "
    "joint signs/IMU frame, motor-side Kd and torque limits, raised-arm entry "
    "and exit transitions are not hardware-validated. Run python -m "
    "tasks.team15_proprio.simulate --mass-kg 0.5 first and "
    "have the laboratory review a separate hardware configuration. "
    "There is deliberately no --allow-hardware bypass."
)


def enforce_simulation_cli(argv: Sequence[str] | None = None) -> None:
    """Reject the official real-robot CLI before it constructs the ROS portal.

    scripts/deploy.py imports the task registry *before* importing the robot
    controller. The policy independently rejects non-MuJoCo controllers.
    This is a mistake-prevention guard, not a sandbox against modified code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    # Mirror all upstream long options so argparse abbreviations and repeated
    # --task options have exactly the same meaning as the official launcher.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
    parser.add_argument("--task")
    parser.add_argument("--list", "-l", action="store_true")
    parser.add_argument("--mujoco", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--exit-mode")
    selected, _ = parser.parse_known_args(args)
    if selected.task == TASK_NAME:
        if not selected.mujoco:
            raise RuntimeError(REAL_ROBOT_BLOCK_REASON)
        raise RuntimeError("This loaded-box task needs its explicit-address simulator. Run python -m tasks.team15_proprio.simulate --mass-kg 0.5 instead.")


def joint_permutation(controller_names: Sequence[str]) -> list[int]:
    """Canonical actor index -> controller index; reject missing/extra names."""
    names = list(controller_names)
    if len(names) != ACTION_DIM or len(set(names)) != ACTION_DIM or set(names) != set(JOINT_NAMES):
        raise ValueError("Expected exactly the 22 named Team15 K1 joints")
    return [names.index(name) for name in JOINT_NAMES]


for _values in (TRAINING_JOINT_NAMES, JOINT_NAMES, DEFAULT_JOINT_POS, ACTION_SCALE,
                SIM_STIFFNESS, SIM_DAMPING, SIM_EFFORT_LIMIT):
    if len(_values) != ACTION_DIM:
        raise ValueError("Invalid Team15 joint constant count")
