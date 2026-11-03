"""Explicit-address free-box extension of the official MuJoCo controller."""
import numpy as np
import torch
import mujoco

from booster_deploy.controllers.base_controller import BaseController
from booster_deploy.controllers.mujoco_controller import MujocoController

from .scene import build_scene, validate_model, ROOT_JOINT


class ProprioHoldMujocoController(MujocoController):
    def __init__(self, cfg, payload_mass_kg=.5):
        # The upstream ctor hardcodes nq=29. Retain its framework/policy/viewer
        # protocol, but initialize and address the added free-body scene here.
        BaseController.__init__(self, cfg)
        self._paused = False
        asset = self._expand_assets_placeholder(self.robot.cfg.mjcf_path)
        self.mj_model, initial, self.scene_audit = build_scene(asset, payload_mass_kg)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.scene_audit["physics_dt_s"] = float(self.mj_model.opt.timestep)
        self.scene_audit["control_dt_s"] = float(self.cfg.policy_dt)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_data.qpos[:] = initial
        self.initial_qpos = initial.copy()
        self.joint_ids = np.array([self.mj_model.joint(n).id for n in cfg.robot.joint_names])
        self.joint_qpos_addresses = self.mj_model.jnt_qposadr[self.joint_ids].copy()
        self.joint_dof_addresses = self.mj_model.jnt_dofadr[self.joint_ids].copy()
        self.motor_ids = np.array([np.flatnonzero(self.mj_model.actuator_trnid[:, 0] == j)[0] for j in self.joint_ids])
        root = self.mj_model.joint(ROOT_JOINT).id
        self.root_qpos_address = int(self.mj_model.jnt_qposadr[root])
        self.root_dof_address = int(self.mj_model.jnt_dofadr[root])
        validate_model(self.mj_model, cfg.robot.joint_names)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        self._ghost_mj_data.qpos[:] = initial
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()
        self._reference_qpos = None
        self.validation_substep_callback = None

    def update_state(self):
        m, d = self.mj_model, self.mj_data
        rq, rv = self.root_qpos_address, self.root_dof_address
        # No slicing of the payload's qpos/qvel into robot observations.
        quat = d.qpos[rq+3:rq+7].copy()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        values = {
            "joint_pos": d.qpos[self.joint_qpos_addresses],
            "joint_vel": d.qvel[self.joint_dof_addresses],
            "feedback_torque": d.qfrc_actuator[self.joint_dof_addresses],
            "root_pos_w": d.qpos[rq:rq+3], "root_quat_w": quat,
            "root_lin_vel_b": rotation.reshape(3,3).T @ d.qvel[rv:rv+3],
            "root_ang_vel_b": d.qvel[rv+3:rv+6],
        }
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite robot state: {name}")
            setattr(self.robot.data, name, torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32,
                                                         device=self.robot.data.device))

    def log_states(self, dof_targets):
        if self.cfg.mujoco.log_states is None:
            return
        if not hasattr(self, "_states"):
            self._states = {k: [] for k in ("qpos", "qvel", "qfrc_actuator", "dof_targets")}
        for key in ("qpos", "qvel", "qfrc_actuator"):
            self._states[key].append(getattr(self.mj_data, key).copy())
        self._states["dof_targets"].append(np.array(dof_targets).copy())
        if len(self._states["qpos"]) % 100 == 0:
            np.savez_compressed(f"{self.cfg.mujoco.log_states}.npz", **{k: np.asarray(v) for k,v in self._states.items()})

    def ctrl_step(self, dof_targets):
        targets = dof_targets.detach().cpu().numpy()
        if targets.shape != (22,) or not np.isfinite(targets).all():
            raise ValueError("Expected 22 finite robot targets")
        self.log_states(targets)
        kp = self.robot.joint_stiffness.cpu().numpy()
        kd = self.robot.joint_damping.cpu().numpy()
        limits = self.robot.effort_limit.cpu().numpy()
        for _ in range(self.decimation):
            q = self.mj_data.qpos[self.joint_qpos_addresses].astype(np.float32)
            v = self.mj_data.qvel[self.joint_dof_addresses].astype(np.float32)
            torque = np.clip(kp*(targets-q)-kd*v, -limits, limits)
            self.mj_data.ctrl[self.motor_ids] = torque
            # XML actuator_forcerange remains independently enforced by MuJoCo.
            mujoco.mj_step(self.mj_model, self.mj_data)
            if self.validation_substep_callback is not None:
                self.validation_substep_callback()

