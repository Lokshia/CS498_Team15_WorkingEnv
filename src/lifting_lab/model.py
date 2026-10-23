"""Physical K1 with a free payload: forearm support or legacy pregrasp.

Forearm support uses only the original robot collision geometry and gravity,
not an attachment. Legacy pregrasp retains its two compliant point connections.
Neither mode welds the robot base to the world.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from booster_mjlab.robots.booster_k1.k1_constants import (
    HOME_KEYFRAME, get_k1_robot_cfg, get_spec,
)

ROBOT_ENTITY = "robot"
PAYLOAD_ENTITY = "payload"
PAYLOAD_BODY = "box"
HAND_SITES = ("left_hand", "right_hand")
GRIP_SITES = ("left_grip", "right_grip")
MIN_PAYLOAD_MASS_KG = 0.001  # MuJoCo requires positive inertia for a free body.
GRIP_HEIGHT_OFFSET_M = 0.03
SUPPORT_MODES = ("pregrasp", "forearms")


def validate_support_mode(mode: str) -> str:
    if mode not in SUPPORT_MODES:
        raise ValueError(f"support_mode must be one of {SUPPORT_MODES}")
    return mode


@dataclass(frozen=True)
class PayloadCfg:
    mass_kg: float = 1.0
    half_size: tuple[float, float, float] = (0.10, 0.20, 0.06)

    def __post_init__(self):
        if not math.isfinite(self.mass_kg) or self.mass_kg < 0:
            raise ValueError("mass_kg must be finite and nonnegative")
        if len(self.half_size) != 3 or any(
            not math.isfinite(x) or x <= 0 for x in self.half_size
        ):
            raise ValueError("half_size must contain three positive finite lengths")

    @property
    def actual_mass_kg(self) -> float:
        """Actual simulated mass; requested zero is explicitly represented by 1 g."""
        return max(self.mass_kg, MIN_PAYLOAD_MASS_KG)

    @property
    def diagonal_inertia(self) -> tuple[float, float, float]:
        x, y, z = self.half_size
        return tuple(self.actual_mass_kg * s / 3 for s in (y*y+z*z, x*x+z*z, x*x+y*y))


def holding_keyframe() -> EntityCfg.InitialStateCfg:
    """Symmetric chest-height pregrasp; all 22 original joints remain actuated."""
    state = deepcopy(HOME_KEYFRAME)
    # Obtained by FK/IK on the pinned K1, then frozen for reproducibility.
    for side, sign in (("Left", 1), ("Right", -1)):
        state.joint_pos.update({
            f"{side}_Shoulder_Pitch": -0.48997132,
            f"{side}_Shoulder_Roll": -sign * 1.55934814,
            f"{side}_Elbow_Pitch": -0.53687104,
            f"{side}_Elbow_Yaw": -sign * 1.66800648,
        })
    return state


def forearm_keyframe() -> EntityCfg.InitialStateCfg:
    """Upper arms down, forearms forward/down 0.06 rad; no base fixture.

    The small downward slope keeps the larger end-ball colliders below the
    box. Its bottom rests on the forearm cylinders, not the two hand balls.
    """
    state = deepcopy(HOME_KEYFRAME)
    for side, sign in (("Left", 1), ("Right", -1)):
        state.joint_pos.update({
            f"{side}_Shoulder_Pitch": 0.0,
            f"{side}_Shoulder_Roll": -sign * math.pi / 2,
            f"{side}_Elbow_Pitch": 0.0,
            f"{side}_Elbow_Yaw": -sign * (math.pi / 2 - 0.06),
        })
    return state


@lru_cache(maxsize=16)
def initial_forearm_geometry(half_size: tuple[float, float, float] = (.10, .20, .06)):
    """Place a level box 1 micrometre above the actual forearm contact surface.

    Use native convex collision distances, not the full cylinder's AABB top:
    the cylinder's highest point can be behind the box's footprint. This is
    initialization only; there are no runtime positional/attachment forces.
    Returned reference sites are diagnostic markers, not gripping contacts.
    """
    PayloadCfg(half_size=half_size)  # Validate dimensions before compiling.
    spec = get_spec()
    box_body = spec.worldbody.add_body(name="support_geometry_probe")
    box_body.add_freejoint(name="support_geometry_probe_joint")
    box_body.add_geom(name="support_geometry_probe_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=half_size, mass=1.0)
    model = spec.compile()
    data = mujoco.MjData(model)
    state = forearm_keyframe()
    data.qpos[:7] = (*state.pos, *state.rot)
    for name, value in state.joint_pos.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    adr = model.jnt_qposadr[model.joint("support_geometry_probe_joint").id]
    center = np.array((.17, 0.0, .8))
    data.qpos[adr:adr+7] = (*center, 1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    box_id = model.geom("support_geometry_probe_box").id
    arms = [model.geom(f"{side}_hand_collision").id for side in ("left", "right")]

    def clearance(z):
        data.qpos[adr + 2] = z
        mujoco.mj_forward(model, data)
        return min(mujoco.mj_geomDistance(model, data, gid, box_id, 1.0, None) for gid in arms)

    low = min(data.geom_xpos[gid, 2] for gid in arms) + half_size[2] - .04
    high = low + .16
    if clearance(low) >= 0 or clearance(high) <= 0:
        raise ValueError("Box dimensions do not overlap the two forearm support surfaces")
    for _ in range(40):
        middle = (low + high) / 2
        if clearance(middle) < 1e-6:
            low = middle
        else:
            high = middle
    center[2] = high
    clearance(high)
    for gid in range(model.ngeom):
        if "collision" in model.geom(gid).name:
            distance = mujoco.mj_geomDistance(model, data, gid, box_id, 1.0, None)
            if distance < -1e-5:
                raise ValueError(f"Initial box intersects {model.geom(gid).name}; adjust box dimensions/pose")
    hands = np.array([data.site_xpos[model.site(name).id] for name in HAND_SITES])
    return tuple(center.tolist()), tuple(tuple(row.tolist()) for row in hands - center)


@lru_cache(maxsize=1)
def initial_grip_geometry() -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return payload world center and its local grip points from exact K1 FK.

    The sites are aligned at the actual frozen joint pose, not rounded target
    coordinates. Consequently the equality constraints start with zero error.
    """
    model = get_spec().compile()
    data = mujoco.MjData(model)
    state = holding_keyframe()
    data.qpos[:7] = (*state.pos, *state.rot)
    for name, value in state.joint_pos.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    hands = np.array([data.site_xpos[model.site(name).id] for name in HAND_SITES])
    center = hands.mean(axis=0) - np.array([0.0, 0.0, GRIP_HEIGHT_OFFSET_M])
    grips = hands - center
    return tuple(center.tolist()), tuple(tuple(row.tolist()) for row in grips)


@lru_cache(maxsize=1)
def known_base_robot_mass() -> float:
    """Mass of the pinned physical robot, excluding payload, in kg."""
    return float(get_spec().compile().body_mass.sum())


robot_mass_kg = known_base_robot_mass


def make_robot_cfg(support_mode: str = "pregrasp") -> EntityCfg:
    validate_support_mode(support_mode)
    state = forearm_keyframe() if support_mode == "forearms" else holding_keyframe()
    cfg = get_k1_robot_cfg(default_keyframe=state)
    cfg.articulation = deepcopy(cfg.articulation)
    # No stochastic actuator-delay draws in the deterministic reference task.
    for actuator in cfg.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 0
        actuator.delay_hold_prob = 0.0
    return cfg


def make_payload_spec(payload: PayloadCfg = PayloadCfg(),
                      support_mode: str = "pregrasp") -> mujoco.MjSpec:
    validate_support_mode(support_mode)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=PAYLOAD_BODY)
    body.add_freejoint(name="payload_free_joint")
    # Explicit body inertia makes the handles massless cosmetic geometry while
    # keeping exactly the requested homogeneous-box inertia (or the 1 g floor).
    body.mass = payload.actual_mass_kg
    body.inertia = payload.diagonal_inertia
    body.explicitinertial = True
    body.add_geom(name="payload_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=payload.half_size, mass=0.0, rgba=(0.85, 0.43, 0.08, 1),
                  contype=1, conaffinity=1, friction=(0.7, 0.005, 0.0001),
                  solref=(0.02, 1), group=0)
    _, grips = (initial_forearm_geometry(payload.half_size) if support_mode == "forearms"
                else initial_grip_geometry())
    for name, point in zip(GRIP_SITES, grips, strict=True):
        # In forearm mode these invisible reference markers carry no force and
        # are NOT contact points. They retain a diagnostic observation layout.
        body.add_site(name=name, pos=point, size=(0.008, 0.008, 0.008),
                      rgba=(0.15, 0.8, 0.3, 0 if support_mode == "forearms" else 1))
        if support_mode == "forearms":
            continue
        sign = 1 if point[1] > 0 else -1
        body.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=(point[0], sign*payload.half_size[1], point[2], *point),
                      size=(0.012, 0, 0), mass=0.0, contype=0, conaffinity=0,
                      rgba=(0.12, 0.12, 0.12, 1))
    return spec


def make_payload_cfg(payload: PayloadCfg = PayloadCfg(),
                     support_mode: str = "pregrasp") -> EntityCfg:
    validate_support_mode(support_mode)
    center, _ = (initial_forearm_geometry(payload.half_size) if support_mode == "forearms"
                 else initial_grip_geometry())
    return EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=center),
                     spec_fn=lambda: make_payload_spec(payload, support_mode))


def add_grip_constraints(spec: mujoco.MjSpec) -> None:
    for side, hand, grip in zip(("left", "right"), HAND_SITES, GRIP_SITES, strict=True):
        spec.add_equality(name=f"{side}_pregrasp", type=mujoco.mjtEq.mjEQ_CONNECT,
                          objtype=mujoco.mjtObj.mjOBJ_SITE,
                          name1=f"robot/{hand}", name2=f"payload/{grip}",
                          solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0))
        # The secured hand is permitted to overlap its handle. Other robot-body,
        # payload, ground, and self collisions remain physically active.
        for body in (f"{side}_hand_link", f"{side}_hand_end_ball"):
            spec.add_exclude(name=f"{body}_grasp_exclusion", bodyname1=f"robot/{body}",
                             bodyname2="payload/box")
