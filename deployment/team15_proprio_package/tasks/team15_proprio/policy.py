"""Proprioceptive hold adapter for a free-box booster_deploy simulation.

By default joint targets are projected into the *deployment XML's* joint
ranges. The next observation uses the action corresponding to the target
actually sent. This is an explicit deployment-side change from the original
training controller; its graph/normalizer is unchanged and must be re-tested.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from booster_deploy.controllers.base_controller import Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.policy_runner import create_policy_runner

from .contract import (
    ACTION_DIM, ACTION_SCALE, CONTROL_DT, DEFAULT_JOINT_POS, JOINT_NAMES,
    OBSERVATION_DIM, REAL_ROBOT_BLOCK_REASON, SOURCE_CHECKPOINT_SHA256, TRAINING_JOINT_NAMES,
    joint_permutation,
)


def _require_mujoco(controller) -> None:
    # The official portal initializes ROS before constructing its Policy.
    # __init__.py's CLI guard prevents that path before portal construction;
    # this independent check also rejects direct non-simulator Policy usage.
    from .simulator import ProprioHoldMujocoController
    if not isinstance(controller, ProprioHoldMujocoController):
        raise RuntimeError(REAL_ROBOT_BLOCK_REASON)


def validate_sim_model(model, controller_names) -> None:
    from .scene import validate_model
    validate_model(model, controller_names)


def load_export_contract(model_path: Path) -> dict:
    contract_path = model_path.with_name("proprio_hold.contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "schema": "team15-proprio-hold-actor-v1",
        "task_schema": "k1-proprioceptive-forearm-hold-v1",
        "source_observation_schema": "k1-imu-proprio-last-action-72-v1",
        "no_vision": True,
        "actor_uses_payload_state": False,
        "mass_observed": False,
        "actor_observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "normalization_included": True,
        "action_output_semantics": "clipped_normalized_joint_position_offset",
        "action_scale_included": False,
        "default_joint_offset_included": False,
        "hardware_ready": False,
        "control_dt_s": CONTROL_DT,
        "history_frames": 1,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"Incompatible export contract {key}: {contract.get(key)!r}")
    if contract.get("action_clip") != [-1, 1]:
        raise ValueError("Expected export action clip [-1, 1]")
    if contract.get("input") != {"name": "obs", "dtype": "float32", "shape": [1, OBSERVATION_DIM]}:
        raise ValueError("Expected a single float32[1,72] obs input")
    if contract.get("output") != {"name": "actions", "dtype": "float32", "shape": [1, ACTION_DIM]}:
        raise ValueError("Expected a single float32[1,22] actions output")
    mapping = contract.get("joint_mapping", {})
    for key, expected in (("training_joint_names", TRAINING_JOINT_NAMES),
                          ("booster_deploy_joint_names", JOINT_NAMES)):
        if mapping.get(key) != list(expected):
            raise ValueError(f"Export canonical joint mapping differs: {key}")
    for key, expected in (("default_joint_pos_rad", DEFAULT_JOINT_POS),
                          ("action_scale_rad", ACTION_SCALE)):
        values = np.asarray(mapping.get(key, []), dtype=float)
        if values.shape != (ACTION_DIM,) or not np.allclose(values, expected, rtol=0., atol=1e-12):
            raise ValueError(f"Export canonical joint mapping differs: {key}")
    hash_key = "onnx_sha256" if model_path.suffix == ".onnx" else "torchscript_sha256"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != contract.get(hash_key):
        raise ValueError("Export file SHA-256 does not match its contract")
    return contract


class ProprioHoldPolicy(Policy):
    def __init__(self, cfg, controller):
        _require_mujoco(controller)
        super().__init__(cfg, controller)
        if not np.isclose(controller.cfg.policy_dt, CONTROL_DT, rtol=0., atol=1e-9):
            raise ValueError("This policy was trained at 50 Hz")
        self.device = torch.device(cfg.device)
        self.robot = controller.robot
        self.robot.data.to(self.device)
        self.actor_to_robot = torch.tensor(joint_permutation(self.robot.cfg.joint_names), dtype=torch.long, device=self.device)
        self.nominal = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=self.device)
        self.scale = torch.tensor(ACTION_SCALE, dtype=torch.float32, device=self.device)
        configured_default = self.robot.default_joint_pos.to(self.device)[self.actor_to_robot]
        if not torch.allclose(configured_default, self.nominal, atol=1e-6, rtol=0.):
            raise ValueError("Default joint pose differs from the model observation/action reference")
        path = Path(cfg.checkpoint_path)
        self.model_path = path if path.is_absolute() else Path(self.task_path) / path
        self.contract = load_export_contract(self.model_path)
        self.runner = create_policy_runner(str(self.model_path), self.device, use_actor_module=False)
        self.last_action = torch.zeros(ACTION_DIM, dtype=torch.float32, device=self.device)
        self.last_effective_action = self.last_action
        self.last_raw_action = torch.zeros_like(self.last_action)
        self.last_target_projection_rad = torch.zeros_like(self.last_action)
        self.max_target_projection_by_joint_rad = torch.zeros_like(self.last_action)
        self.projected_target_steps = 0
        self.max_target_projection_rad = 0.
        self.target_lower = self.target_upper = None
        self._model_layout_validated = False

    def reset(self):
        _require_mujoco(self.controller)
        validate_sim_model(self.controller.mj_model, self.robot.cfg.joint_names)
        self.configure_target_limits(self.controller.mj_model)
        self._model_layout_validated = True
        self.last_action.zero_()
        self.last_raw_action.zero_()
        self.last_target_projection_rad.zero_()
        self.max_target_projection_by_joint_rad.zero_()
        self.projected_target_steps = 0
        self.max_target_projection_rad = 0.

    def configure_target_limits(self, model):
        """Read ranges by canonical joint name, never enlarge the asset limits."""
        limits = []
        for name in JOINT_NAMES:
            joint = model.joint(name)
            if bool(model.jnt_limited[joint.id]):
                lo, hi = map(float, model.jnt_range[joint.id])
                if not np.isfinite([lo, hi]).all() or lo >= hi:
                    raise ValueError(f"Invalid XML joint limits for {name}")
                limits.append((lo, hi))
            else:
                limits.append((-float("inf"), float("inf")))
        ranges = torch.tensor(limits, dtype=torch.float32, device=self.device)
        self.target_lower, self.target_upper = ranges[:, 0], ranges[:, 1]
        if torch.any(self.nominal < self.target_lower) or torch.any(self.nominal > self.target_upper):
            raise ValueError("The reference pose lies outside deployment XML limits")

    def projection_statistics(self):
        """Plain JSON-ready audit data; every reset starts a new episode."""
        return {
            "enabled": bool(self.cfg.project_joint_targets),
            "projected_target_steps": self.projected_target_steps,
            "max_target_projection_rad": self.max_target_projection_rad,
            "max_target_projection_by_joint_rad": dict(zip(
                JOINT_NAMES, self.max_target_projection_by_joint_rad.detach().cpu().tolist())),
            "previous_action_semantics": "effective action from sent joint target",
        }

    def compute_observation(self):
        data = self.robot.data
        vectors = ((data.root_ang_vel_b, 3), (data.root_quat_w, 4),
                   (data.joint_pos, ACTION_DIM), (data.joint_vel, ACTION_DIM))
        for value, size in vectors:
            if tuple(value.shape) != (size,) or not torch.isfinite(value).all():
                raise ValueError("Invalid/non-finite simulator state; refusing inference")
        quat = data.root_quat_w.to(dtype=torch.float32, device=self.device)
        if abs(float(torch.linalg.vector_norm(quat)) - 1.) > .001:
            raise ValueError("Expected normalized wxyz root quaternion")
        gravity = lab_math.quat_apply_inverse(quat, torch.tensor([0., 0., -1.], device=self.device))
        if float(gravity[2]) > -np.cos(np.deg2rad(45.)):
            self.controller.stop()
            raise RuntimeError("Team15 simulation stopped: trunk tilt exceeded 45 degrees")
        obs = torch.cat((data.root_ang_vel_b, gravity,
                         data.joint_pos[self.actor_to_robot] - self.nominal,
                         data.joint_vel[self.actor_to_robot], self.last_action)).to(dtype=torch.float32, device=self.device)
        if obs.shape != (OBSERVATION_DIM,) or not torch.isfinite(obs).all():
            raise ValueError("Expected finite 72-dimensional observation")
        return obs

    def action_to_targets(self, action):
        if tuple(action.shape) != (ACTION_DIM,) or not torch.isfinite(action).all():
            raise ValueError("Expected exactly 22 finite actions")
        # Defensive clip matches training; the exported actor already clips.
        action = action.to(device=self.device, dtype=torch.float32).clamp(-1., 1.)
        actor_targets = self.nominal + self.scale * action
        self.last_raw_action.copy_(action)
        projection = torch.zeros_like(action)
        effective_action = action
        if self.cfg.project_joint_targets:
            if self.target_lower is None or self.target_upper is None:
                raise RuntimeError("Call reset() to load deployment XML joint limits first")
            projected = torch.minimum(torch.maximum(actor_targets, self.target_lower), self.target_upper)
            projection = projected - actor_targets
            changed = projection != 0
            # Preserve the exact original floating-point action for joints
            # that did not need projection; do not introduce roundoff drift.
            effective_action = torch.where(changed, (projected - self.nominal) / self.scale, action)
            actor_targets = projected
            if bool(changed.any()):
                self.projected_target_steps += 1
                correction = projection.abs()
                self.max_target_projection_rad = max(self.max_target_projection_rad, float(correction.max()))
                self.max_target_projection_by_joint_rad.copy_(torch.maximum(self.max_target_projection_by_joint_rad, correction))
        self.last_target_projection_rad.copy_(projection)
        robot_targets = torch.empty_like(actor_targets)
        robot_targets[self.actor_to_robot] = actor_targets
        self.last_action.copy_(effective_action)
        return robot_targets

    def inference(self):
        _require_mujoco(self.controller)
        if not self._model_layout_validated:
            raise RuntimeError("Call controller.start()/policy.reset() before inference")
        with torch.inference_mode():
            actions = self.runner(self.compute_observation().unsqueeze(0))
        if tuple(actions.shape) != (1, ACTION_DIM):
            raise ValueError("Model must return actions with shape [1,22]")
        return self.action_to_targets(actions[0])


@configclass
class ProprioHoldPolicyCfg(PolicyCfg):
    constructor = ProprioHoldPolicy
    checkpoint_path: str = "models/proprio_hold.onnx"
    # False is only for reproducing the original simulation audit baseline.
    # It does not enable hardware mode; both guards still reject that path.
    project_joint_targets: bool = True
