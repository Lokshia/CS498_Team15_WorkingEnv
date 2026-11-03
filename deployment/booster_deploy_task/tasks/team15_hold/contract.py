"""Frozen actor conventions for the native K1 task; no camera or box-state inputs.

Simulation gains and XML target bounds are task overrides, not hardware validation.
"""
from __future__ import annotations
import math
from collections.abc import Sequence
TASK_NAME = "k1_team15_hold"
UPSTREAM_COMMIT = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"
CONTROL_DT = .02
OBSERVATION_DIM = 72
ACTION_DIM = 22
MODEL_SHA256 = "c75c00f943ddca0a94ed6722a9848a2443909150aaffc63d253f184b4b55b921"
CONTRACT_SHA256 = "c22b6e5138c5ce085afebab39304cf4649e02d1bd39b78bd15c0f9a50107e2d7"
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
# Frozen simulation reference: official booster_assets 12a9751 K1 geometry is not the training asset. With
# training q_default, root z=.5125 penetrates both foot boxes by .005573700934m.
# Lift by that clearance plus 0.1 mm; no force or pose constraint is added.
SIM_INIT_POS = (0., 0., .5181737009342727)
SOURCE_CHECKPOINT_SHA256 = "b0722baf7d79d513d54cb8afec74860ac4aed5296bbd8049d73ec9e4afcc1382"


# Frozen deployment XML ranges; these are not a motor-side torque protection.
TARGET_RANGES_RAD = ((-1.012, 1.012), (-0.314, 0.794), (-2.932, 1.196), (-1.642, 1.629), (-1.885, 1.885), (-2.242, 0.816), (-2.932, 1.196), (-1.642, 1.629), (-1.885, 1.885), (-0.816, 2.242), (-2.958, 2.226), (-0.375, 1.536), (-1.012, 1.012), (0.0, 2.321), (-0.87, 0.345), (-0.345, 0.345), (-2.958, 2.226), (-1.536, 0.375), (-1.012, 1.012), (0.0, 2.321), (-0.87, 0.345), (-0.345, 0.345))

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
