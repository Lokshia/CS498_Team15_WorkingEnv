"""Simulation controller configuration; gains reproduce training parameters.

The upstream asset and torque controller are intentionally tested as supplied.
They are not assumed dynamically equivalent to mjlab's implicit position
actuators and integration settings.

The adapter additionally projects targets into this asset's XML joint limits,
and feeds the resulting effective previous action back to the model. This
explicit controller change is enabled by default and has its own test report.
"""
from booster_deploy.controllers.controller_cfg import (
    BoosterRobotControllerCfg, ControllerCfg, MujocoControllerCfg, PrepareStateCfg,
)
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass

from .contract import (
    CONTROL_DT, DEFAULT_JOINT_POS, JOINT_NAMES, SIM_DAMPING, SIM_EFFORT_LIMIT,
    SIM_INIT_POS, SIM_STIFFNESS,
)
from .policy import ProprioHoldPolicyCfg


@configclass
class ProprioHoldControllerCfg(ControllerCfg):
    policy_dt = CONTROL_DT
    robot = K1_CFG.replace(
        joint_names=list(JOINT_NAMES),
        # BaseController's qpos and motor arrays use serial order. The actor
        # explicitly maps by name; do not use locomotion's interleaved list.
        sim_joint_names=list(JOINT_NAMES),
        default_joint_pos=list(DEFAULT_JOINT_POS),
        joint_stiffness=list(SIM_STIFFNESS),
        joint_damping=list(SIM_DAMPING),
        effort_limit=list(SIM_EFFORT_LIMIT),
        prepare_mode="standing",
        prepare_state=PrepareStateCfg(
            joint_pos=list(DEFAULT_JOINT_POS),
            stiffness=list(SIM_STIFFNESS),
            damping=list(SIM_DAMPING),
        ),
    )
    policy = ProprioHoldPolicyCfg()
    vel_command = None
    mujoco = MujocoControllerCfg(init_pos=list(SIM_INIT_POS), decimation=4)
    # Unreachable in the supported package, which rejects real-robot startup.
    booster = BoosterRobotControllerCfg(exit_mode="damping")
