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


@dataclass(kw_only=True)
class LiftingEnvCfg(ManagerBasedRlEnvCfg):
    support_mode: str = "pregrasp"
    requested_payload_mass_kg: float = 1.0
    actual_payload_mass_kg: float = 1.0
    reference_height_m: float = 0.61
    mass_observed: bool = True
    reset_xy_range_m: float = 0.05
    reset_yaw_range_rad: float = 0.05
    reset_speed_range_m_s: float = 0.02
    support_grace_steps: int = 6  # 0.12 s at 50 Hz, including initial settling.


def make_env_cfg(*, payload_mass_kg: float = 1.0, num_envs: int = 64,
                 episode_length_s: float = 20.0, mass_observed: bool = True,
                 reference_height_m: float | None = None, play: bool = False,
                 reset_xy_range_m: float = 0.05, reset_yaw_range_rad: float = 0.05,
                 reset_speed_range_m_s: float = 0.02,
                 support_mode: str = "pregrasp") -> LiftingEnvCfg:
    """Create an independent, reproducible environment with fixed mass per batch.

    A mass sweep constructs a fresh environment for each mass, recompiling all
    inertial constants. Reference height defaults to the exact pregrasp FK.
    No startup randomization, pushes, command resampling, or frozen posture
    reward is used. All 22 joints act; the base is free; corrective steps are
    allowed. ``play`` does not weaken fall checks or change the finite horizon.
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
    return LiftingEnvCfg(
        support_mode=support_mode,
        requested_payload_mass_kg=payload.mass_kg,
        actual_payload_mass_kg=payload.actual_mass_kg,
        reference_height_m=reference, mass_observed=mass_observed,
        reset_xy_range_m=reset_xy_range_m, reset_yaw_range_rad=reset_yaw_range_rad,
        reset_speed_range_m_s=reset_speed_range_m_s,
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
        events={
            "reset": EventTermCfg(func=mdp.reset_pregrasp, mode="reset", params={
                "xy_range": reset_xy_range_m, "yaw_range": reset_yaw_range_rad,
                "speed_range": reset_speed_range_m_s,
            }),
            **({"support_history": EventTermCfg(func=mdp.update_support_state, mode="step")}
               if support_mode == "forearms" else {}),
        },
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
