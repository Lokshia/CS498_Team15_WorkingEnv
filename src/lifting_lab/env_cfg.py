"""Deterministic chest-height payload holding task for CPU/GPU batched mjlab."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as builtin
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from booster_mjlab.robots.booster_k1.k1_constants import K1_ACTION_SCALE
from booster_mjlab.terrains.contact import terrain_collisions

from . import mdp
from .model import (PayloadCfg, add_grip_constraints, initial_grip_geometry,
                    initial_forearm_geometry, make_payload_cfg, make_robot_cfg,
                    validate_support_mode)

FOREARM_SUPPORT_GEOMS = ("left_hand_collision", "right_hand_collision")
ALLOWED_AUXILIARY_GEOMS = ("left_hand_end_collision", "right_hand_end_collision")
MAX_PAYLOAD_XY_JITTER_M = 0.01
MAX_PAYLOAD_YAW_JITTER_RAD = 0.03


@dataclass(kw_only=True)
class LiftingEnvCfg(ManagerBasedRlEnvCfg):
    support_mode: str = "pregrasp"
    requested_payload_mass_kg: float = 1.0
    actual_payload_mass_kg: float = 1.0
    payload_half_size_m: tuple[float, float, float] = (0.10, 0.20, 0.06)
    reference_height_m: float = 0.61
    mass_observed: bool = True
    reset_xy_range_m: float = 0.05
    reset_yaw_range_rad: float = 0.05
    reset_speed_range_m_s: float = 0.02
    support_grace_steps: int = 6  # 0.12 s at 50 Hz, including initial settling.
    # Optional per-environment randomization. The defaults below reproduce the
    # fixed-mass, no-push task exactly; see make_env_cfg for the semantics.
    payload_mass_range_kg: tuple[float, float] | None = None
    payload_com_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    payload_xy_jitter_m: float = 0.0
    payload_yaw_jitter_rad: float = 0.0
    push_interval_s: tuple[float, float] | None = None
    push_velocity_m_s: float = 0.0


def _finite_nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return float(value)


def make_env_cfg(*, payload_mass_kg: float = 1.0, num_envs: int = 64,
                 episode_length_s: float = 20.0, mass_observed: bool = True,
                 reference_height_m: float | None = None, play: bool = False,
                 reset_xy_range_m: float = 0.05, reset_yaw_range_rad: float = 0.05,
                 reset_speed_range_m_s: float = 0.02,
                 support_mode: str = "pregrasp",
                 payload_mass_range_kg: tuple[float, float] | None = None,
                 payload_com_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 payload_xy_jitter_m: float = 0.0, payload_yaw_jitter_rad: float = 0.0,
                 push_interval_s: tuple[float, float] | None = None,
                 push_velocity_m_s: float = 0.0) -> LiftingEnvCfg:
    """Create an independent, reproducible environment with a nominal mass per batch.

    A mass sweep constructs a fresh environment for each mass, recompiling all
    inertial constants. Reference height defaults to the exact pregrasp FK.
    No startup randomization, command resampling, or frozen posture reward is
    used. All 22 joints act; the base is free; corrective steps are allowed.
    ``play`` does not weaken fall checks or change the finite horizon.

    Everything below is off by default and leaves the compiled model untouched:

    - ``payload_mass_range_kg``: each reset draws the box mass uniformly from
      this range and writes it, with the matching homogeneous-box inertia,
      into that environment's Warp model. ``payload_mass_kg`` stays the nominal
      value reported in metadata. Requested masses below 1 g use the 1 g floor.
    - ``payload_com_offset_m``: half-ranges of a uniform centre-of-mass shift
      in the box frame, like a ballast that is not centred. The collision box
      is unchanged, so the shift must stay inside the box.
    - ``payload_xy_jitter_m`` / ``payload_yaw_jitter_rad``: additional box-only
      pose offset relative to the forearms at reset: x forward in ``[0, j]``
      (the nominal box is 1 cm from the trunk), y and yaw symmetric.
    - ``push_interval_s`` + ``push_velocity_m_s``: horizontal root-velocity
      kicks on the robot with per-environment intervals (interval event).
    """
    del play
    validate_support_mode(support_mode)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be positive and finite")
    for value in (reset_xy_range_m, reset_yaw_range_rad, reset_speed_range_m_s):
        if not math.isfinite(value) or value < 0:
            raise ValueError("reset ranges must be nonnegative and finite")
    payload = PayloadCfg(mass_kg=payload_mass_kg)
    if payload_mass_range_kg is not None:
        low, high = (_finite_nonnegative(v, "payload_mass_range_kg") for v in payload_mass_range_kg)
        if low > high:
            raise ValueError("payload_mass_range_kg must be ordered (low, high)")
        payload_mass_range_kg = (low, high)
    if len(payload_com_offset_m) != 3:
        raise ValueError("payload_com_offset_m needs three half-ranges")
    payload_com_offset_m = tuple(_finite_nonnegative(v, "payload_com_offset_m") for v in payload_com_offset_m)
    if any(offset >= half for offset, half in zip(payload_com_offset_m, payload.half_size, strict=True)):
        raise ValueError("payload_com_offset_m must keep the centre of mass inside the box")
    payload_xy_jitter_m = _finite_nonnegative(payload_xy_jitter_m, "payload_xy_jitter_m")
    payload_yaw_jitter_rad = _finite_nonnegative(payload_yaw_jitter_rad, "payload_yaw_jitter_rad")
    if payload_xy_jitter_m > MAX_PAYLOAD_XY_JITTER_M or payload_yaw_jitter_rad > MAX_PAYLOAD_YAW_JITTER_RAD:
        # Beyond 1 cm forward the box rests on the hand-end balls and the
        # zero-action start no longer reaches two-cylinder support in 6 steps.
        raise ValueError("payload jitter exceeds the range verified to keep the box on both forearm cylinders")
    push_velocity_m_s = _finite_nonnegative(push_velocity_m_s, "push_velocity_m_s")
    if push_interval_s is not None:
        low, high = push_interval_s
        if not (math.isfinite(low) and math.isfinite(high)) or low <= 0 or low > high:
            raise ValueError("push_interval_s must be positive, finite and ordered (low, high)")
        push_interval_s = (float(low), float(high))
    if (push_interval_s is None) != (push_velocity_m_s == 0):
        raise ValueError("Pushes need both push_interval_s and a positive push_velocity_m_s")
    initial_center, _ = (initial_forearm_geometry(payload.half_size) if support_mode == "forearms"
                         else initial_grip_geometry())
    reference = initial_center[2] if reference_height_m is None else reference_height_m
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("reference_height_m must be positive and finite")
    terms = {"state": ObservationTermCfg(func=mdp.task_observation,
                                         params={"mass_observed": mass_observed})}
    sensors = [ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_foot_link|right_foot_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
    )]
    if support_mode == "forearms":
        # Exact original cylinder geoms: hand-end balls are NOT forearms.
        for name, primary in (
            ("forearm_payload_contact", ContactMatch(
                mode="geom", pattern=r"^(left_hand_collision|right_hand_collision)$", entity="robot")),
            ("forbidden_robot_payload_contact", ContactMatch(
                # Unnamed K1 visual meshes have no collision bits; do not
                # create duplicate empty-name sensors for those meshes.
                mode="geom", pattern=".+", entity="robot",
                exclude=FOREARM_SUPPORT_GEOMS + ALLOWED_AUXILIARY_GEOMS)),
            ("ground_payload_contact", ContactMatch(mode="body", pattern="terrain")),
        ):
            sensors.append(ContactSensorCfg(
                name=name, primary=primary,
                secondary=ContactMatch(mode="geom", pattern="payload_box", entity="payload"),
                fields=("found", "force"), reduce="netforce", num_slots=1,
                # Detect forbidden loaded contacts even if they disappear
                # before the final substep. No extra simulator forces added.
                history_length=4 if name != "forearm_payload_contact" else 0,
            ))
    events = {
        "reset": EventTermCfg(func=mdp.reset_pregrasp, mode="reset", params={
            "xy_range": reset_xy_range_m, "yaw_range": reset_yaw_range_rad,
            "speed_range": reset_speed_range_m_s,
            "payload_xy_jitter": payload_xy_jitter_m, "payload_yaw_jitter": payload_yaw_jitter_rad,
        }),
    }
    if payload_mass_range_kg is not None or any(payload_com_offset_m):
        events["payload_inertial"] = EventTermCfg(func=mdp.randomize_payload_inertial, mode="reset", params={
            "mass_range_kg": payload_mass_range_kg, "com_offset_m": payload_com_offset_m,
        })
    if support_mode == "forearms":
        events["support_history"] = EventTermCfg(func=mdp.update_support_state, mode="step")
    if push_interval_s is not None:
        speed = push_velocity_m_s
        events["push"] = EventTermCfg(func=builtin.push_by_setting_velocity, mode="interval",
                                      interval_range_s=push_interval_s, params={
                                          "velocity_range": {"x": (-speed, speed), "y": (-speed, speed)},
                                          "asset_cfg": SceneEntityCfg("robot"),
                                      })
    return LiftingEnvCfg(
        support_mode=support_mode,
        requested_payload_mass_kg=payload.mass_kg,
        actual_payload_mass_kg=payload.actual_mass_kg,
        payload_half_size_m=tuple(payload.half_size),
        reference_height_m=reference, mass_observed=mass_observed,
        reset_xy_range_m=reset_xy_range_m, reset_yaw_range_rad=reset_yaw_range_rad,
        reset_speed_range_m_s=reset_speed_range_m_s,
        payload_mass_range_kg=payload_mass_range_kg, payload_com_offset_m=payload_com_offset_m,
        payload_xy_jitter_m=payload_xy_jitter_m, payload_yaw_jitter_rad=payload_yaw_jitter_rad,
        push_interval_s=push_interval_s, push_velocity_m_s=push_velocity_m_s,
        scene=SceneCfg(
            num_envs=num_envs, env_spacing=2.5,
            entities={"robot": make_robot_cfg(support_mode),
                      "payload": make_payload_cfg(payload, support_mode)},
            terrain=TerrainEntityCfg(terrain_type="plane", collisions=terrain_collisions()),
            spec_fn=add_grip_constraints if support_mode == "pregrasp" else None,
            sensors=tuple(sensors),
        ),
        observations={
            name: ObservationGroupCfg(terms=deepcopy(terms), concatenate_terms=True, enable_corruption=False)
            for name in ("actor", "critic")
        },
        actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                                     scale=deepcopy(K1_ACTION_SCALE), use_default_offset=True)},
        events=events,
        rewards={
            "hold_height": RewardTermCfg(func=mdp.hold_height_reward, weight=3.0),
            "payload_level": RewardTermCfg(func=mdp.payload_level_reward, weight=2.0),
            "upright": RewardTermCfg(func=mdp.upright_reward, weight=2.0),
            "balance_approx": RewardTermCfg(func=mdp.balance_reward, weight=0.5),
            "effort": RewardTermCfg(func=mdp.effort_cost, weight=-0.00002),
            "smooth_actions": RewardTermCfg(func=builtin.action_rate_l2, weight=-0.02),
            "payload_motion": RewardTermCfg(func=mdp.payload_motion_cost, weight=-0.05),
            "foot_slip": RewardTermCfg(func=mdp.foot_slip_cost, weight=-0.2),
            "joint_limits": RewardTermCfg(func=builtin.joint_pos_limits, weight=-0.1),
            **({"supported_hold": RewardTermCfg(func=mdp.supported_hold_reward, weight=3.0)}
               if support_mode == "forearms" else {}),
        },
        terminations={
            "time_out": TerminationTermCfg(func=builtin.time_out, time_out=True),
            "fell": TerminationTermCfg(func=mdp.fell),
            **({"grip_failed": TerminationTermCfg(func=mdp.grip_failed)}
               if support_mode == "pregrasp" else {
                   "support_lost": TerminationTermCfg(func=mdp.support_lost),
                   "payload_disallowed_contact": TerminationTermCfg(func=mdp.payload_disallowed_contact),
               }),
            "nonfinite": TerminationTermCfg(func=builtin.nan_detection),
        },
        sim=SimulationCfg(nconmax=100, njmax=600, contact_sensor_maxmatch=128,
                          mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=20,
                                           gravity=(0.0, 0.0, -9.81), ccd_iterations=60)),
        decimation=4, episode_length_s=episode_length_s,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="robot", body_name="Trunk", distance=2.4,
                            elevation=-12.0, azimuth=135.0),
    )
