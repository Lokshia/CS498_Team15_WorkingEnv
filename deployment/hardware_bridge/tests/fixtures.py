"""Synthetic software-test profiles. NEVER a configuration for real motors."""
from copy import deepcopy
import json
from pathlib import Path

from team15_hardware.config import CONFIRMATIONS, MODEL_SHA, PROTOCOL, UPSTREAM_COMMIT


def load_contract():
    root = Path(__file__).resolve().parents[1]
    path = root.parent / "team15_proprio_package/tasks/team15_proprio/models/proprio_hold.contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_config(contract):
    nominal = list(contract["joint_mapping"]["default_joint_pos_rad"])
    names = contract["joint_mapping"]["booster_deploy_joint_names"]
    return {
        "schema": "team15-hardware-site-v1",
        "deployment_target": "simulation_only",
        "upstream_booster_deploy_commit": UPSTREAM_COMMIT,
        "model": {"onnx_sha256": MODEL_SHA, "control_dt_s": .02,
                  "observation_dim": 72, "action_dim": 22},
        "review": {"robot_serial": "SYNTHETIC-NO-REAL-ROBOT", "reviewer": "software test fixture",
                   "evidence": "Synthetic software test only; no physical calibration was performed.",
                   **{key: True for key in CONFIRMATIONS}},
        "protocol": deepcopy(PROTOCOL),
        "transport": {"low_state_topic": "/low_state", "joint_ctrl_topic": "/joint_ctrl",
                      "rpc_service": "/booster_rpc_service", "rpc_timeout_s": .1,
                      "motor_weight": [0.] * 22},
        "imu": {"convention": "rpy_xyz_imu_to_world_rad", "gyro_frame": "imu",
                "body_from_imu": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]},
        "joints": [{"name": name, "serial_index": i, "sign": 1., "zero_offset_rad": 0.,
                    "model_min_rad": -3., "model_max_rad": 3., "kp": 10., "kd": 1.,
                    "target_rate_limit_rad_s": 2., "measured_speed_limit_rad_s": 10.,
                    "measured_torque_limit_nm": 100., "tracking_error_limit_rad": 1.}
                   for i, name in enumerate(names)],
        "limits": {"state_timeout_s": .1, "command_timeout_s": .1,
                   "inference_timeout_s": .01, "mode_timeout_s": .5,
                   "prime_settle_s": .05, "prepare_timeout_s": 3., "ready_timeout_s": 2.,
                   "run_timeout_s": 2., "max_trunk_tilt_deg": 20.,
                   "subscriber_timeout_s": .5, "mode_check_interval_s": .05,
                   "status_rpc_timeout_s": .1},
        "prepare": {"start_model_q": nominal.copy(), "start_tolerance_rad": .02,
                    "ready_tolerance_rad": .02, "ready_dwell_s": .05,
                    "ready_speed_limit_rad_s": .1,
                    "waypoints": [{"model_q": nominal.copy(), "duration_s": 1.}]},
        "stop": {"exit_mode": "damping", "firmware_watchdog_timeout_s": .1,
                 "procedure": "Synthetic exit event only; no real robot or stop procedure."},
    }
