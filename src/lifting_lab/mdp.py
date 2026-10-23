"""Batched torch observations, balance terms, and measured hold diagnostics.

All kinematics come from the simulator. The support margin is a deliberately
labelled *geometric approximation*: an axis-aligned envelope around both sole
centers, not a contact-force support polygon or a dynamic stability certificate.
"""
from __future__ import annotations

import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply, quat_from_euler_xyz, quat_mul
from mjlab.envs.mdp.events import reset_scene_to_default, resolve_env_ids

from .model import GRIP_SITES, HAND_SITES, robot_mass_kg


def reset_pregrasp(env, env_ids, xy_range: float = 0.05,
                  yaw_range: float = 0.05, speed_range: float = 0.02):
    """Seeded common transform/velocity preserves the initial relative geometry.

    All joints retain the exact initial pregrasp. No independent arm jitter,
    mass randomization or pushes. The tiny common XY velocity varies episodes.
    Set all three ranges to zero for a completely identical reset baseline.
    """
    env_ids = resolve_env_ids(env, env_ids)
    reset_scene_to_default(env, env_ids)
    n = len(env_ids)
    xy = (2 * torch.rand(n, 2, device=env.device) - 1) * xy_range
    yaw = (2 * torch.rand(n, device=env.device) - 1) * yaw_range
    zero = torch.zeros_like(yaw)
    rotation = quat_from_euler_xyz(zero, zero, yaw)
    velocity = (2 * torch.rand(n, 2, device=env.device) - 1) * speed_range
    for name in ("robot", "payload"):
        entity = env.scene[name]
        state = entity.data.default_root_state[env_ids].clone()
        state[:, :3] = quat_apply(rotation, state[:, :3])
        state[:, :3] += env.scene.env_origins[env_ids]
        state[:, :2] += xy
        state[:, 3:7] = quat_mul(rotation, state[:, 3:7])
        state[:, 7:9] = velocity
        entity.write_root_state_to_sim(state, env_ids)
    if getattr(env.cfg, "support_mode", "pregrasp") == "forearms":
        reset_support_state(env, env_ids)


def _ids(env):
    if not hasattr(env, "_lifting_site_ids"):
        env._lifting_site_ids = (
            env.scene["robot"].find_sites(HAND_SITES, preserve_order=True)[0],
            env.scene["payload"].find_sites(GRIP_SITES, preserve_order=True)[0],
            env.scene["robot"].find_sites(("left_foot_sole", "right_foot_sole"),
                                          preserve_order=True)[0],
        )
    return env._lifting_site_ids


def payload_height(env):
    return env.scene["payload"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def height_error(env):
    return payload_height(env) - env.cfg.reference_height_m


def payload_upright(env):
    return -env.scene["payload"].data.projected_gravity_b[:, 2]


def robot_upright(env):
    return -env.scene["robot"].data.projected_gravity_b[:, 2]


def combined_com_w(env):
    """Mass-weighted inertial COM of robot + payload from xipos and body_mass.

    Ground/world bodies have zero mass. This task has no other massive entities.
    Reading the simulation model also handles per-world mass tensors correctly.
    """
    masses = env.sim.model.body_mass
    if masses.ndim == 1:
        masses = masses.unsqueeze(0)
    return (env.sim.data.xipos * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-9)


def com_margin_approx(env):
    # Conservative sole half-extents, in world axes. Rotated/tilted soles and
    # airborne feet make this unsuitable as an exact support-area measurement.
    soles = env.scene["robot"].data.site_pos_w[:, _ids(env)[2], :2]
    half = soles.new_tensor((0.07, 0.025))
    lower, upper = soles.amin(dim=1) - half, soles.amax(dim=1) + half
    com = combined_com_w(env)[:, :2]
    return torch.minimum(com - lower, upper - com).amin(dim=-1)


def grip_displacement(env):
    """Legacy grasp residual; forearms: nominal relative-pose offset, not contact."""
    hand_ids, grip_ids, _ = _ids(env)
    return (env.scene["robot"].data.site_pos_w[:, hand_ids]
            - env.scene["payload"].data.site_pos_w[:, grip_ids])


def task_observation(env, mass_observed: bool = True):
    """75 robot features + 22 payload/task features; blind masks only mass.

    Both variants retain the same shape. Payload/robot velocities, orientation,
    grip errors and proprioception still allow implicit mass inference.
    """
    robot = env.scene["robot"].data
    payload = env.scene["payload"].data
    q = robot.root_link_quat_w
    position_b = quat_apply_inverse(q, payload.root_link_pos_w - robot.root_link_pos_w)
    velocity_b = quat_apply_inverse(q, payload.root_link_lin_vel_w - robot.root_link_lin_vel_w)
    angular_b = quat_apply_inverse(q, payload.root_link_ang_vel_w)
    grip_b = quat_apply_inverse(q[:, None, :].expand(-1, 2, -1), grip_displacement(env))
    observed_mass = env.cfg.actual_payload_mass_kg / robot_mass_kg() if mass_observed else 0.0
    return torch.cat((
        robot.root_link_ang_vel_b, robot.projected_gravity_b,
        robot.joint_pos - robot.default_joint_pos, robot.joint_vel,
        env.action_manager.action, robot.root_link_lin_vel_b,
        position_b, velocity_b, payload.projected_gravity_b, angular_b,
        height_error(env)[:, None], grip_b.flatten(1),
        torch.full((env.num_envs, 1), observed_mass, device=env.device),
        (robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])[:, None],
        payload_height(env)[:, None],
    ), dim=-1)


def hold_height_reward(env, std: float = 0.05):
    return torch.exp(-(height_error(env) / std).square())


def payload_level_reward(env, std: float = 0.20):
    return torch.exp(-(1.0 - payload_upright(env).clamp(-1, 1)) / (std * std))


def upright_reward(env):
    return robot_upright(env).clamp(0, 1).square()


def balance_reward(env):
    grounded = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs, -1).gt(0).any(-1)
    return torch.exp(-torch.relu(-com_margin_approx(env)).square() / 0.04**2) * grounded


def foot_slip_cost(env):
    """Squared world-XY sole velocity only while that foot contacts the floor."""
    velocity = env.scene["robot"].data.site_lin_vel_w[:, _ids(env)[2], :2]
    contact = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs, -1).gt(0)
    return (velocity.square().sum(-1) * contact).sum(-1)


def effort_cost(env):
    return env.scene["robot"].data.qfrc_actuator.square().sum(dim=-1)


def payload_motion_cost(env):
    data = env.scene["payload"].data
    return data.root_link_lin_vel_w.square().sum(-1) + 0.1 * data.root_link_ang_vel_w.square().sum(-1)


def fell(env):
    robot = env.scene["robot"].data
    low = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < 0.28
    return low | (robot_upright(env) < math.cos(math.radians(60))) | (payload_height(env) < 0.20)


def grip_failed(env):
    return torch.linalg.vector_norm(grip_displacement(env), dim=-1).amax(-1) > 0.15


_SUPPORT_BOOL_KEYS = (
    "left_forearm_contact", "right_forearm_contact", "payload_forearm_supported",
    "payload_disallowed_contact", "support_lost",
)


def _support_state(env):
    if not hasattr(env, "_lifting_support_state"):
        env._lifting_support_state = {
            key: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for key in _SUPPORT_BOOL_KEYS
        }
        env._lifting_support_state.update({
            key: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            for key in ("forearm_contact_count", "support_gap_steps")
        })
        env._lifting_support_state["forearm_support_force_n"] = torch.zeros(
            env.num_envs, device=env.device)
        env._lifting_support_state["step"] = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device)
    return env._lifting_support_state


def reset_support_state(env, env_ids):
    """Clear only reset rows, even when episode_length_buf has not reset yet."""
    for key, values in _support_state(env).items():
        values[env_ids] = -1 if key == "step" else 0


def _forbidden_contact(sensor_data, num_envs):
    present = sensor_data.found.reshape(num_envs, -1).gt(0).any(-1)
    history = sensor_data.force_history
    if history is not None:
        # Current geometric contact also counts when its constraint force is
        # zero. History catches transient loaded collisions during decimation.
        present = present | history.square().sum(-1).gt(1e-12).flatten(1).any(-1)
    return present


def forearm_contact_metrics(env) -> dict[str, torch.Tensor]:
    """Read this control step's contact sample without advancing any timers.

    MuJoCo netforce points primary -> secondary. With a robot geom primary
    and payload secondary, force[..., 2] is world +Z force ON the payload,
    not contact-frame normal force. Both exact cylinders must apply >1e-6 N
    upward; mere proximity, lateral pushing, or one-arm support do not qualify.

    Termination/reward use the last physics-substep sample (up to 0.005 s
    before the integrated state). The step event commits that same sample for
    evaluation, so post-forward reads cannot disagree with termination.
    """
    state = _support_state(env)
    sensor = env.scene["forearm_payload_contact"]
    contact = sensor.data.found.reshape(env.num_envs, 2).gt(0)
    force_z = sensor.data.force.reshape(env.num_envs, 2, 3)[..., 2]
    supported = (contact & (force_z > 1e-6)).all(-1)
    forbidden = _forbidden_contact(env.scene["forbidden_robot_payload_contact"].data, env.num_envs)
    forbidden |= _forbidden_contact(env.scene["ground_payload_contact"].data, env.num_envs)
    gap = torch.where(supported, 0, state["support_gap_steps"] + 1)
    gap = torch.where(env.episode_length_buf > 0, gap, 0)
    # Do not assume regex expansion order is left/right.
    left = sensor.primary_names.index("left_hand_collision")
    right = sensor.primary_names.index("right_hand_collision")
    values = {
        "left_forearm_contact": contact[:, left],
        "right_forearm_contact": contact[:, right],
        "forearm_contact_count": contact.sum(-1),
        "forearm_support_force_n": force_z.sum(-1),
        "payload_forearm_supported": supported,
        "payload_disallowed_contact": forbidden,
        "support_gap_steps": gap,
        "support_lost": gap > env.cfg.support_grace_steps,
    }
    committed = state["step"] == env.episode_length_buf
    return {key: torch.where(committed, state[key], value) for key, value in values.items()}


def update_support_state(env, env_ids=None):
    """Once-per-control-step event, after RL signals and before auto-reset.

    Repeated calls at the same episode step are idempotent. Reset explicitly
    invalidates the selected rows, rather than relying on a global sim clock.
    """
    del env_ids
    values = forearm_contact_metrics(env)
    state = _support_state(env)
    for key, value in values.items():
        state[key].copy_(value)
    state["step"].copy_(env.episode_length_buf)


def support_lost(env):
    return forearm_contact_metrics(env)["support_lost"]


def payload_disallowed_contact(env):
    return forearm_contact_metrics(env)["payload_disallowed_contact"]


def supported_hold_reward(env):
    values = forearm_contact_metrics(env)
    supported = values["payload_forearm_supported"] & ~values["payload_disallowed_contact"]
    return supported * hold_height_reward(env) * payload_level_reward(env)


def task_metrics(env) -> dict[str, torch.Tensor]:
    """Instantaneous diagnostics. Complete-episode success is accumulated outside.

    valid_hold is NOT an episode-success flag; every sample of a full-duration
    episode must satisfy its posture/height criteria, with no fall. Forearm
    contact loss alone permits six consecutive control steps (including startup).
    """
    height = payload_height(env)
    error = height_error(env)
    tilt = torch.rad2deg(torch.acos(payload_upright(env).clamp(-1, 1)))
    contacts = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs, -1) > 0
    offset_key = ("support_reference_offset_m" if getattr(env.cfg, "support_mode", "pregrasp") == "forearms"
                  else "grip_error_m")
    is_fallen = fell(env)
    nonfinite = (~torch.isfinite(env.sim.data.qpos).all(-1)
                 | ~torch.isfinite(env.sim.data.qvel).all(-1))
    valid = (tilt < 10.0) & (error.abs() < 0.05) & ~is_fallen & ~nonfinite
    result = {
        "payload_height_m": height,
        "height_error_m": error,
        "payload_tilt_deg": tilt,
        "robot_tilt_deg": torch.rad2deg(torch.acos(robot_upright(env).clamp(-1, 1))),
        "upright": robot_upright(env),
        "com_margin_approx_m": com_margin_approx(env),
        "base_height_m": env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2],
        offset_key: torch.linalg.vector_norm(grip_displacement(env), dim=-1).amax(-1),
        "torque_rms_nm": env.scene["robot"].data.qfrc_actuator.square().mean(-1).sqrt(),
        "feet_contact_count": contacts.sum(-1).float(),
        "foot_slip_m2_s2": foot_slip_cost(env),
        "fell": is_fallen,
        "nonfinite": nonfinite,
    }
    if getattr(env.cfg, "support_mode", "pregrasp") == "forearms":
        result.update(forearm_contact_metrics(env))
        valid &= ~result["payload_disallowed_contact"] & ~result["support_lost"]
    result["valid_hold"] = valid.float()
    return result


def metric(env, key: str):
    return task_metrics(env)[key]
