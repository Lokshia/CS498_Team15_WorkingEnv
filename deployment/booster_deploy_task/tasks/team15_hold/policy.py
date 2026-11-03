"""Native booster_deploy Policy using only IMU and serial joint state.

Returned targets and inference logs are not publisher acknowledgments. The
upstream RobotData has no receive timestamp; this task cannot measure freshness.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from booster_deploy.controllers.base_controller import Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.policy_runner import create_policy_runner

from .contract import (
    ACTION_DIM, ACTION_SCALE, CONTROL_DT, DEFAULT_JOINT_POS, JOINT_NAMES,
    OBSERVATION_DIM, TARGET_RANGES_RAD, CONTRACT_SHA256, UPSTREAM_COMMIT, joint_permutation,
)
from .validation import load_export_contract


class Team15HoldPolicy(Policy):
    def __init__(self, cfg, controller):
        super().__init__(cfg, controller)
        if not np.isclose(controller.cfg.policy_dt, CONTROL_DT, rtol=0., atol=1e-9):
            raise ValueError("Team15 actor requires policy_dt=0.02 seconds")
        if not np.isfinite(cfg.start_pose_tolerance_rad) or cfg.start_pose_tolerance_rad <= 0:
            raise ValueError("Start pose tolerance must be positive and finite")
        if not cfg.project_joint_targets:
            raise ValueError("Native task requires the frozen target projection")
        self.device = torch.device(cfg.device)
        self.robot = controller.robot
        self.robot.data.to(self.device)
        self.actor_to_robot = torch.tensor(joint_permutation(self.robot.cfg.joint_names), dtype=torch.long, device=self.device)
        self.nominal = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=self.device)
        self.scale = torch.tensor(ACTION_SCALE, dtype=torch.float32, device=self.device)
        if not torch.allclose(self.robot.default_joint_pos.to(self.device)[self.actor_to_robot], self.nominal, atol=1e-6, rtol=0.):
            raise ValueError("Configured nominal pose differs from exported actor")
        path = Path(cfg.checkpoint_path)
        self.model_path = path if path.is_absolute() else Path(self.task_path) / path
        self.contract = load_export_contract(self.model_path)
        self.runner = create_policy_runner(str(self.model_path), self.device, use_actor_module=False)
        self.last_action = torch.zeros(ACTION_DIM, dtype=torch.float32, device=self.device)
        self.last_effective_action = self.last_action
        self.last_raw_action = torch.zeros_like(self.last_action)
        self.last_target_projection_rad = torch.zeros_like(self.last_action)
        self.max_target_projection_by_joint_rad = torch.zeros_like(self.last_action)
        ranges = torch.tensor(TARGET_RANGES_RAD, dtype=torch.float32, device=self.device)
        self.target_lower, self.target_upper = ranges[:, 0], ranges[:, 1]
        self.projected_target_steps = 0
        self.max_target_projection_rad = 0.
        self._ready = False
        self._step = 0
        self._log = None
        self.log_path = None

    def _write_log(self, row):
        if self._log is not None:
            row = dict(row)
            row.setdefault("wall_time_ns", time.time_ns())
            row.setdefault("monotonic_s", time.monotonic())
            self._log.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            self._log.flush()

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None

    def _stop_for_fault(self, exc):
        self._ready = False
        try:
            self._write_log({"event":"fault", "step":self._step,
                             "error":str(exc), "monotonic_s":time.monotonic()})
        finally:
            try:
                self.close()
            finally:
                self.controller.stop()

    def reset(self):
        self._ready = False
        self.close()
        self.last_action.zero_()
        self.last_raw_action.zero_()
        self.last_target_projection_rad.zero_()
        self.max_target_projection_by_joint_rad.zero_()
        self.projected_target_steps = 0
        self.max_target_projection_rad = 0.
        self._step = 0
        try:
            # Upstream run() calls update_state before reset. No fabricated timestamp.
            self.compute_observation()
            error = (self.robot.data.joint_pos[self.actor_to_robot] - self.nominal).abs()
            if float(error.max()) > self.cfg.start_pose_tolerance_rad:
                joint = JOINT_NAMES[int(error.argmax())]
                raise ValueError(f"Initial pose differs from holding reference: {joint}, {float(error.max()):.3f} rad")
            directory = Path(os.environ.get("TEAM15_LOG_DIR", "logs/team15_hold"))
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.log_path = directory / f"inference_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}.jsonl"
            self._log = self.log_path.open("x", encoding="utf-8")
            gains = {"stiffness":list(self.robot.cfg.joint_stiffness),
                     "damping":list(self.robot.cfg.joint_damping),
                     "effort_limit":list(self.robot.cfg.effort_limit)}
            gain_hash = hashlib.sha256(json.dumps(gains, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self._write_log({"event":"session", "schema":"team15-native-inference-v1",
                             "model_sha256":self.contract["onnx_sha256"],
                             "contract_sha256":CONTRACT_SHA256,
                             "run_gain_sha256":gain_hash, "run_gains":gains,
                             "run_gain_source":"Frozen Team15 simulation SIM_STIFFNESS/SIM_DAMPING/SIM_EFFORT_LIMIT task overrides",
                             "prepare_gain_source":f"Official K1_CFG at {UPSTREAM_COMMIT}; nominal holding pose is a task override; 1-second transition not hardware-validated",
                             "policy_dt_s":CONTROL_DT, "joint_names":list(JOINT_NAMES),
                             "controller_joint_names":list(self.robot.cfg.joint_names),
                             "task_configuration":{
                                 "prepare_mode":self.robot.cfg.prepare_mode,
                                 "prepare_joint_pos_rad":list(self.robot.cfg.prepare_state.joint_pos),
                                 "prepare_stiffness":list(self.robot.cfg.prepare_state.stiffness),
                                 "prepare_damping":list(self.robot.cfg.prepare_state.damping),
                                 "nominal_joint_pos_rad":list(self.robot.cfg.default_joint_pos),
                                 "target_ranges_rad":[list(row) for row in TARGET_RANGES_RAD],
                                 "exit_mode":self.controller.cfg.booster.exit_mode,
                                 "policy_dt_s":self.controller.cfg.policy_dt},
                             "hardware_validated":False, "state_receive_age_available":False,
                             "clock_note":"Wall time is local system clock for bag alignment if clocks are synchronized; neither wall nor monotonic time is a sensor timestamp",
                             "log_semantics":"inference outputs before ctrl_step; not publish or motor acknowledgment",
                             "start_pose_tolerance_rad":self.cfg.start_pose_tolerance_rad})
            self._ready = True
        except Exception as exc:
            self._stop_for_fault(exc)
            raise

    def compute_observation(self):
        data = self.robot.data
        vectors = ((data.root_ang_vel_b, 3), (data.root_quat_w, 4),
                   (data.joint_pos, ACTION_DIM), (data.joint_vel, ACTION_DIM))
        for value, size in vectors:
            if tuple(value.shape) != (size,) or not torch.isfinite(value).all():
                raise ValueError("Invalid/nonfinite robot state; refusing inference")
        quat = data.root_quat_w.to(dtype=torch.float32, device=self.device)
        if abs(float(torch.linalg.vector_norm(quat)) - 1.) > .001:
            raise ValueError("Expected normalized wxyz root quaternion")
        gravity = lab_math.quat_apply_inverse(quat, torch.tensor([0., 0., -1.], device=self.device))
        if float(gravity[2]) > -np.cos(np.deg2rad(45.)):
            raise RuntimeError("Team15 stopped: trunk tilt exceeded 45 degrees")
        obs = torch.cat((data.root_ang_vel_b, gravity,
                         data.joint_pos[self.actor_to_robot] - self.nominal,
                         data.joint_vel[self.actor_to_robot], self.last_action)).to(dtype=torch.float32, device=self.device)
        if obs.shape != (OBSERVATION_DIM,) or not torch.isfinite(obs).all():
            raise ValueError("Expected finite 72-dimensional observation")
        return obs

    def action_to_targets(self, action):
        if tuple(action.shape) != (ACTION_DIM,) or not torch.isfinite(action).all():
            raise ValueError("Expected exactly 22 finite actions")
        action = action.to(device=self.device, dtype=torch.float32).clamp(-1., 1.)
        self.last_raw_action.copy_(action)
        actor_targets = self.nominal + self.scale * action
        projected = torch.minimum(torch.maximum(actor_targets, self.target_lower), self.target_upper)
        projection = projected - actor_targets
        changed = projection != 0
        effective = torch.where(changed, (projected - self.nominal) / self.scale, action)
        if bool(changed.any()):
            self.projected_target_steps += 1
            correction = projection.abs()
            self.max_target_projection_rad = max(self.max_target_projection_rad, float(correction.max()))
            self.max_target_projection_by_joint_rad.copy_(torch.maximum(self.max_target_projection_by_joint_rad, correction))
        self.last_target_projection_rad.copy_(projection)
        robot_targets = torch.empty_like(projected)
        robot_targets[self.actor_to_robot] = projected
        self.last_action.copy_(effective)
        return robot_targets

    def projection_statistics(self):
        return {"enabled":True, "projected_target_steps":self.projected_target_steps,
                "max_target_projection_rad":self.max_target_projection_rad,
                "max_target_projection_by_joint_rad":dict(zip(JOINT_NAMES, self.max_target_projection_by_joint_rad.detach().cpu().tolist())),
                "previous_action_semantics":"effective action from previously returned joint target; not publish acknowledgment"}

    def inference(self):
        try:
            if not self._ready:
                raise RuntimeError("Call controller.start()/policy.reset() before inference")
            observation = self.compute_observation()
            with torch.inference_mode():
                actions = self.runner(observation.unsqueeze(0))
            if tuple(actions.shape) != (1, ACTION_DIM):
                raise ValueError("Model must return actions with shape [1,22]")
            raw = actions[0].detach().clone()
            target = self.action_to_targets(raw)
            self._step += 1
            self._write_log({"event":"inference", "step":self._step, "monotonic_s":time.monotonic(),
                             "observation":observation.detach().cpu().tolist(),
                             "raw_action":raw.cpu().tolist(),
                             "target_controller_order_rad":target.detach().cpu().tolist(),
                             "effective_action":self.last_action.detach().cpu().tolist()})
            return target
        except Exception as exc:
            # Upstream ctrl_step follows inference immediately. Never return a
            # target after stop(): raising prevents that remaining call.
            self._stop_for_fault(exc)
            raise


@configclass
class Team15HoldPolicyCfg(PolicyCfg):
    constructor = Team15HoldPolicy
    checkpoint_path: str = "models/proprio_hold.onnx"
    # Task-side start check, not a measured factory safety limit. It is reached
    # AFTER upstream X preparation; it does not protect the 1-second raise.
    start_pose_tolerance_rad: float = .25
    # Compatibility field for frozen simulation audit; projection is mandatory.
    project_joint_targets: bool = True
