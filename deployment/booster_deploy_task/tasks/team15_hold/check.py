"""Offline: python -m tasks.team15_hold.check (no ROS or robot connection)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from booster_deploy.controllers.base_controller import BoosterRobot
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.registry import get_task
from .contract import (TASK_NAME, UPSTREAM_COMMIT, JOINT_NAMES, DEFAULT_JOINT_POS,
                       SIM_STIFFNESS, SIM_DAMPING, SIM_EFFORT_LIMIT, TARGET_RANGES_RAD,
                       CONTRACT_SHA256, CONTROL_DT)
from .policy import Team15HoldPolicy
from .validation import load_export_contract


class OfflineController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = BoosterRobot(cfg.robot)
        self.stopped = False
    def stop(self):
        self.stopped = True


def check():
    cfg = get_task(TASK_NAME).copy()
    cfg.policy.device = "cpu"
    if cfg.policy_dt != CONTROL_DT or cfg.vel_command is not None:
        raise ValueError("Task timing/command configuration differs")
    if cfg.robot.name != "Booster_K1" or cfg.robot.prepare_mode != "standing" or cfg.booster.exit_mode != "damping":
        raise ValueError("Task robot/prepare/exit defaults differ")
    for actual, expected in (
        (cfg.robot.joint_names, JOINT_NAMES), (cfg.robot.sim_joint_names, JOINT_NAMES),
        (cfg.robot.default_joint_pos, DEFAULT_JOINT_POS),
        (cfg.robot.joint_stiffness, SIM_STIFFNESS), (cfg.robot.joint_damping, SIM_DAMPING),
        (cfg.robot.effort_limit, SIM_EFFORT_LIMIT),
        (cfg.robot.prepare_state.stiffness, K1_CFG.prepare_state.stiffness),
        (cfg.robot.prepare_state.damping, K1_CFG.prepare_state.damping),
        (cfg.robot.prepare_state.joint_pos, DEFAULT_JOINT_POS),
    ):
        if list(actual) != list(expected):
            raise ValueError("Task joint/gain/prepare configuration differs")
    if "rclpy" in sys.modules:
        raise RuntimeError("Offline checker unexpectedly imported ROS")
    models = Path(__file__).with_name("models")
    contract = load_export_contract(models/"proprio_hold.onnx")
    vector_path = models/"proprio_hold.test_vectors.npz"
    if hashlib.sha256(vector_path.read_bytes()).hexdigest() != contract["test_vectors_sha256"]:
        raise ValueError("Test vector bytes differ from frozen export")
    source = json.loads((models/"target_envelope.sources.json").read_text(encoding="utf-8"))
    if source["joint_names"] != list(JOINT_NAMES) or source["joint_ranges_rad"] != [list(row) for row in TARGET_RANGES_RAD]:
        raise ValueError("Projection bounds differ from their frozen source receipt")
    with np.load(vector_path, allow_pickle=False) as vectors:
        observations, expected = vectors["observations"], vectors["expected_actions"]
    if observations.shape != (534,72) or expected.shape != (534,22):
        raise ValueError("Expected all 534 frozen export vectors")
    if not np.isfinite(observations).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite test vectors")
    controller = OfflineController(cfg)
    policy = Team15HoldPolicy(cfg.policy, controller)
    actual = []
    with torch.inference_mode():
        for row in observations:
            output = policy.runner(torch.tensor(row, dtype=torch.float32)[None])
            if output.shape != (1,22) or not torch.isfinite(output).all():
                raise ValueError("Nonfinite/invalid actor output")
            actual.append(output[0].numpy().copy())
            targets = policy.action_to_targets(output[0])
            if not torch.isfinite(targets).all() or torch.any(targets < policy.target_lower) or torch.any(targets > policy.target_upper):
                raise ValueError("Projection produced out-of-envelope targets")
    actual = np.stack(actual)
    maximum = float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Actor differs from frozen vectors: {maximum}")
    return {"task":TASK_NAME, "offline_check_passed":True, "test_vectors":534,
            "max_absolute_error":maximum, "onnx_sha256":contract["onnx_sha256"],
            "contract_sha256":CONTRACT_SHA256, "upstream_commit":UPSTREAM_COMMIT,
            "observation_dim":72, "action_dim":22, "policy_dt_s":CONTROL_DT,
            "no_vision":True, "no_payload_state_input":True,
            "prepare_mode":cfg.robot.prepare_mode, "default_exit_mode":cfg.booster.exit_mode,
            "run_gains_are_task_overrides":True,
            "hardware_tested":False, "ros_imported":False,
            "limits":"Offline export/config check only; no real robot commands or validation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = check()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
