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
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from .model import GRIP_SITES, HAND_SITES, MIN_PAYLOAD_MASS_KG, robot_mass_kg

# The nominal box rests 1 cm in front of the trunk collider, so jitter only
# moves it forward in x (uniform [0, j]) and sideways/yawed symmetrically.
# A yawed box meets the forearm cylinders up to 0.3 mm above the nominal rest
# height (mj_geomDistance, default box). A 0.2 mm lift keeps that initial
# overlap within the contact softness; a larger drop lets an off-centre box
# land on one cylinder and rock onto the hand balls instead of both cylinders.
PAYLOAD_JITTER_LIFT_M = 0.0002


def _uniform(n, scale, device, columns=None):
    shape = (n,) if columns is None else (n, columns)
    return (2 * torch.rand(shape, device=device) - 1) * scale


def reset_pregrasp(env, env_ids, xy_range: float = 0.05,
                  yaw_range: float = 0.05, speed_range: float = 0.02,
                  payload_xy_jitter: float = 0.0, payload_yaw_jitter: float = 0.0):
    """Seeded common transform/velocity preserves the initial relative geometry.

    All joints retain the exact initial pregrasp; there is no independent arm
    jitter. The tiny common XY velocity varies episodes. Set all three ranges
    to zero for a completely identical reset baseline. The optional payload
    jitter moves only the box, in the yawed robot frame, so the policy does
    not see one fixed box placement on the forearms.
    """
    env_ids = resolve_env_ids(env, env_ids)
    reset_scene_to_default(env, env_ids)
    n = len(env_ids)
    xy = _uniform(n, xy_range, env.device, 2)
    yaw = _uniform(n, yaw_range, env.device)
    zero = torch.zeros_like(yaw)
    rotation = quat_from_euler_xyz(zero, zero, yaw)
    velocity = _uniform(n, speed_range, env.device, 2)
    jittered = payload_xy_jitter > 0 or payload_yaw_jitter > 0
    for name in ("robot", "payload"):
        entity = env.scene[name]
        state = entity.data.default_root_state[env_ids].clone()
        if name == "payload" and jittered:
            shift = torch.cat((torch.rand(n, 1, device=env.device) * payload_xy_jitter,
                               _uniform(n, payload_xy_jitter, env.device, 1),
                               torch.full((n, 1), PAYLOAD_JITTER_LIFT_M, device=env.device)), dim=-1)
            state[:, :3] += shift
            state[:, 3:7] = quat_mul(quat_from_euler_xyz(zero, zero, _uniform(n, payload_yaw_jitter, env.device)),
                                     state[:, 3:7])
        state[:, :3] = quat_apply(rotation, state[:, :3])
        state[:, :3] += env.scene.env_origins[env_ids]
        state[:, :2] += xy
        state[:, 3:7] = quat_mul(rotation, state[:, 3:7])
        state[:, 7:9] = velocity
        entity.write_root_state_to_sim(state, env_ids)
    if getattr(env.cfg, "support_mode", "pregrasp") == "forearms":
        reset_support_state(env, env_ids)


def _payload_body_id(env):
    if not hasattr(env, "_lifting_payload_body_id"):
        env._lifting_payload_body_id = int(env.scene["payload"].indexing.body_ids[0])
    return env._lifting_payload_body_id


def payload_mass_kg(env):
    """Per-environment simulated box mass, read back from the Warp model."""
    masses = env.sim.model.body_mass[:, _payload_body_id(env)]
    return masses.expand(env.num_envs) if masses.shape[0] == 1 else masses


@requires_model_fields("body_mass", "body_ipos", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_payload_inertial(env, env_ids, mass_range_kg=None, com_offset_m=(0.0, 0.0, 0.0)):
    """Draw each reset environment's box mass and centre of mass.

    The collision box is unchanged; only the inertial parameters of
    ``payload/box`` are written per world, like a ballast placed off-centre.
    Mass is uniform in ``mass_range_kg`` (nominal mass when None, 1 g floor),
    the inertial origin is uniform within ``+-com_offset_m`` in the box frame,
    and the inertia tensor stays the homogeneous-box value about that origin,
    scaled by the drawn mass. The event manager then runs ``set_const`` so
    subtree masses and constraint inverse weights match the new values.
    """
    env_ids = resolve_env_ids(env, env_ids)
    n = len(env_ids)
    if mass_range_kg is None:
        mass = torch.full((n,), env.cfg.actual_payload_mass_kg, device=env.device)
    else:
        low, high = mass_range_kg
        mass = (torch.rand(n, device=env.device) * (high - low) + low).clamp_min(MIN_PAYLOAD_MASS_KG)
    x, y, z = env.cfg.payload_half_size_m
    unit_inertia = mass.new_tensor((y * y + z * z, x * x + z * z, x * x + y * y)) / 3
    body = _payload_body_id(env)
    model = env.sim.model
    model.body_mass[env_ids, body] = mass
    model.body_ipos[env_ids, body] = _uniform(n, mass.new_tensor(com_offset_m), env.device, 3)
    model.body_inertia[env_ids, body] = mass[:, None] * unit_inertia


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
    if mass_observed:
        observed_mass = (payload_mass_kg(env) / robot_mass_kg()).to(q.dtype)[:, None]
    else:
        observed_mass = torch.zeros((env.num_envs, 1), device=env.device, dtype=q.dtype)
    return torch.cat((
        robot.root_link_ang_vel_b, robot.projected_gravity_b,
        robot.joint_pos - robot.default_joint_pos, robot.joint_vel,
        env.action_manager.action, robot.root_link_lin_vel_b,
        position_b, velocity_b, payload.projected_gravity_b, angular_b,
        height_error(env)[:, None], grip_b.flatten(1),
        observed_mass,
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


def unsupported_cost(env):
    """Penalize every unsupported sample, including the permitted contact grace."""
    return (~forearm_contact_metrics(env)["payload_forearm_supported"]).float()


def missing_feet_cost(env):
    """Zero/two points for both/no feet on the floor; stepping is not terminated."""
    contact = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs, 2).gt(0)
    return (~contact).sum(-1).float()


def robot_vertical_velocity_cost(env):
    return env.scene["robot"].data.root_link_lin_vel_w[:, 2].square()


def robot_angular_velocity_cost(env):
    return env.scene["robot"].data.root_link_ang_vel_w.square().sum(-1)


def joint_velocity_cost(env):
    return env.scene["robot"].data.joint_vel.square().sum(-1)


def stance_foot_data(env):
    """Left/right sole normals and ground reactions, in world coordinates.

    netforce sensors report primary -> secondary: feet -> terrain is downward,
    so negate it for the reaction ON the robot. No assumed regex ordering.
    Normal +Z is the sole site's local normal (the K1 sites have no rotation).
    """
    sensor = env.scene["feet_ground_contact"]
    if not hasattr(env, "_stance_contact_ids"):
        names = [name.split("/")[-1] for name in sensor.primary_names]
        expected = ("left_foot_link", "right_foot_link")
        if len(names) != 2 or set(names) != set(expected):
            raise ValueError(f"Expected two named foot contact primaries, got {names}")
        env._stance_contact_ids = [names.index(name) for name in expected]
    ids = env._stance_contact_ids
    forces = -sensor.data.force.reshape(env.num_envs, 2, 3)[:, ids]
    contact = sensor.data.found.reshape(env.num_envs, 2)[:, ids] > 0
    quat = env.scene["robot"].data.site_quat_w[:, _ids(env)[2]]
    z = torch.zeros_like(quat[..., :3])
    z[..., 2] = 1
    normal = quat_apply(quat, z)
    return normal[..., 2].clamp(-1, 1), forces, contact


def feet_flat_cost(env):
    """Zero for upward level soles, positive for tilt including inverted feet.

    Applied even in flight; the separate missing-feet cost prevents hovering
    from satisfying the stance objective. This is not a contact-area metric.
    """
    normal_z, _, _ = stance_foot_data(env)
    return (1.0 - normal_z).sum(-1)


def foot_load_share_cost(env, minimum_share: float = 0.30):
    """Soft 30--70% deadband, not a demand for exact equal foot loading.

    Missing or tensile contacts cannot earn a balanced-load score. Total force
    must exceed 10% of weight; transient dynamics remain allowed with a cost.
    """
    _, force, contact = stance_foot_data(env)
    vertical = force[..., 2].clamp_min(0) * contact
    total = vertical.sum(-1)
    share = vertical / total[:, None].clamp_min(1e-6)
    cost = (torch.relu(minimum_share - share) / minimum_share).square().sum(-1)
    weight = (robot_mass_kg() + payload_mass_kg(env)) * 9.81
    return torch.where(total > 0.1 * weight, cost, torch.ones_like(cost))


def leg_posture_cost(env, scale_rad: float = 0.25):
    """Soft preference for the nominal bent-knee leg pose; arms remain free."""
    robot = env.scene["robot"]
    if not hasattr(env, "_stance_leg_ids"):
        env._stance_leg_ids = robot.find_joints(
            ("Left_Hip_.*", "Left_Knee_Pitch", "Left_Ankle_.*",
             "Right_Hip_.*", "Right_Knee_Pitch", "Right_Ankle_.*"),
            preserve_order=True)[0]
        if len(env._stance_leg_ids) != 12:
            raise ValueError("stance profile requires the twelve named K1 leg joints")
    displacement = (robot.data.joint_pos - robot.data.default_joint_pos)[:, env._stance_leg_ids]
    return (displacement / scale_rad).square().mean(-1)


def trunk_tilt_cost(env, scale_deg: float = 10.0):
    """Yaw-invariant upright cost: one at scale_deg, including inverted poses."""
    if not math.isfinite(scale_deg) or not 0 < scale_deg <= 180:
        raise ValueError("scale_deg must be finite and in (0, 180]")
    scale = 1.0 - math.cos(math.radians(scale_deg))
    return (1.0 - robot_upright(env).clamp(-1.0, 1.0)) / scale


def leg_symmetry_cost(env, deadband_deg: float = 5.0, scale_deg: float = 10.0):
    """Soft mirrored leg preference; small asymmetry is free and arms stay free.

    Pitch and knee axes have equal joint signs; roll/yaw axes have opposite
    signs across the sagittal mirror plane. The deadband permits load-dependent
    adjustments rather than requiring identical leg trajectories.
    """
    if not math.isfinite(deadband_deg) or deadband_deg < 0:
        raise ValueError("deadband_deg must be finite and nonnegative")
    if not math.isfinite(scale_deg) or scale_deg <= 0:
        raise ValueError("scale_deg must be finite and positive")
    robot = env.scene["robot"]
    if not hasattr(env, "_stance_mirror_joint_ids"):
        joints = ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
        sides = []
        for side in ("Left", "Right"):
            expected = tuple(f"{side}_{joint}" for joint in joints)
            indices, names = robot.find_joints(expected, preserve_order=True)
            if len(indices) != 6 or tuple(names) != expected:
                raise ValueError(f"stance symmetry requires ordered named {side} leg joints")
            sides.append(indices)
        env._stance_mirror_joint_ids = tuple(sides)
    left_ids, right_ids = env._stance_mirror_joint_ids
    positions = robot.data.joint_pos
    signs = positions.new_tensor((1.0, -1.0, -1.0, 1.0, 1.0, -1.0))
    residual = positions[:, left_ids] - signs * positions[:, right_ids]
    excess = torch.relu(residual.abs() - math.radians(deadband_deg))
    return (excess / math.radians(scale_deg)).square().mean(-1)


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
        "joint_velocity_rms_rad_s": env.scene["robot"].data.joint_vel.square().mean(-1).sqrt(),
        "action_delta_rms": (env.action_manager.action - env.action_manager.prev_action).square().mean(-1).sqrt(),
        "robot_vertical_speed_m_s": env.scene["robot"].data.root_link_lin_vel_w[:, 2].abs(),
        "robot_angular_speed_rad_s": torch.linalg.vector_norm(env.scene["robot"].data.root_link_ang_vel_w, dim=-1),
        "payload_linear_speed_m_s": torch.linalg.vector_norm(env.scene["payload"].data.root_link_lin_vel_w, dim=-1),
        "payload_angular_speed_rad_s": torch.linalg.vector_norm(env.scene["payload"].data.root_link_ang_vel_w, dim=-1),
        "both_feet_contact": contacts.all(-1).float(),
        "airborne": (~contacts.any(-1)).float(),
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
