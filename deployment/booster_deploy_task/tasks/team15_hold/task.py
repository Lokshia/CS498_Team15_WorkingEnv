"""Native task config; original upstream portal owns ROS and mode transitions."""
from booster_deploy.controllers.controller_cfg import (
    BoosterRobotControllerCfg, ControllerCfg, MujocoControllerCfg, PrepareStateCfg,
)
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from .contract import (CONTROL_DT, DEFAULT_JOINT_POS, JOINT_NAMES, SIM_DAMPING,
                       SIM_EFFORT_LIMIT, SIM_INIT_POS, SIM_STIFFNESS)
from .policy import Team15HoldPolicyCfg


@configclass
class Team15HoldControllerCfg(ControllerCfg):
    policy_dt = CONTROL_DT
    robot = K1_CFG.replace(
        joint_names=list(JOINT_NAMES), sim_joint_names=list(JOINT_NAMES),
        default_joint_pos=list(DEFAULT_JOINT_POS),
        # These override official walking gains with the frozen holding
        # simulation gains. Real-world holding has not yet been validated.
        joint_stiffness=list(SIM_STIFFNESS), joint_damping=list(SIM_DAMPING),
        effort_limit=list(SIM_EFFORT_LIMIT),
        prepare_mode="standing",
        prepare_state=PrepareStateCfg(
            joint_pos=list(DEFAULT_JOINT_POS),
            stiffness=list(K1_CFG.prepare_state.stiffness),
            damping=list(K1_CFG.prepare_state.damping),
        ),
    )
    policy = Team15HoldPolicyCfg()
    vel_command = None
    mujoco = MujocoControllerCfg(init_pos=list(SIM_INIT_POS), decimation=4)
    # Damping releases active posture support; use a physical support/TA.
    # Original deploy.py --exit-mode may override this default.
    booster = BoosterRobotControllerCfg(exit_mode="damping")
