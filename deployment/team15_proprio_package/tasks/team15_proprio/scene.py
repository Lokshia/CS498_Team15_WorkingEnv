"""In-memory overlay: unmodified official K1 body/motors plus one free box.

Collision meshes remain the official ones. Explicit box/forearm contact pairs
set effective sliding friction to the training value .7; no weld, attachment,
support force, disabled body collision, or pose reset is applied at runtime.
"""
from pathlib import Path
import hashlib
import math

import mujoco
import numpy as np

from .contract import DEFAULT_JOINT_POS, JOINT_NAMES, SIM_INIT_POS, joint_permutation

PAYLOAD_BODY = "team15_payload"
PAYLOAD_GEOM = "team15_payload_box"
PAYLOAD_JOINT = "team15_payload_free"
ROOT_JOINT = "world_joint"
SUPPORT_GEOMS = ("team15_left_forearm_contact", "team15_right_forearm_contact")
FOOT_GEOMS = ("team15_left_foot_contact", "team15_right_foot_contact")
HALF_SIZE = np.array((.10, .20, .06), dtype=float)


def validate_model(model, controller_names):
    """Check addressed robot joints, unit-gear torque motors, and free payload."""
    names = list(controller_names)
    joint_permutation(names)
    if (model.nq, model.nv, model.nu, model.njnt) != (36, 34, 22, 24):
        raise ValueError("Expected robot free base + 22 hinges + one free box")
    if model.neq:
        raise ValueError("Holding scene must not have equality attachments")
    ids = np.array([model.joint(n).id for n in names])
    if not np.all(model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Robot joints must be hinges")
    for root in (ROOT_JOINT, PAYLOAD_JOINT):
        if model.jnt_type[model.joint(root).id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{root} must be free")
    if not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("Expected joint torque motors")
    if set(model.actuator_trnid[:, 0].tolist()) != set(ids.tolist()):
        raise ValueError("Exactly one motor per robot joint is required; box must be unactuated")
    if not np.allclose(model.actuator_gear[:, 0], 1.) or not np.allclose(model.actuator_gear[:, 1:], 0.):
        raise ValueError("Expected unit-gear motors")
    if (not np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
            or not np.allclose(model.actuator_gainprm[:, 0], 1.)
            or not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)):
        raise ValueError("Expected explicit torque motors, not position servos")


def build_scene(asset_xml, mass_kg=.5):
    """Return (model, initial_qpos, audit); never write the vendor XML."""
    if not math.isfinite(mass_kg) or mass_kg < .001:
        raise ValueError("Free payload mass must be finite and at least 0.001 kg")
    asset_xml = Path(asset_xml).resolve(strict=True)
    before = hashlib.sha256(asset_xml.read_bytes()).hexdigest()
    spec = mujoco.MjSpec.from_file(str(asset_xml))
    # Name previously anonymous geometry without altering its physical values.
    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"team15_official_geom_{i}"
    for side, support, foot in zip(("left", "right"), SUPPORT_GEOMS, FOOT_GEOMS):
        arm_body = spec.body(f"{side}_elbow_yaw_link")
        arm_geoms = [g for g in arm_body.geoms if g.contype or g.conaffinity]
        foot_body = spec.body(f"{side}_ankle_roll_link")
        foot_geoms = [g for g in foot_body.geoms if g.contype or g.conaffinity]
        if len(arm_geoms) != 1 or len(foot_geoms) != 1:
            raise ValueError("Official arm/sole collision layout changed")
        arm_geoms[0].name = support
        foot_geoms[0].name = foot
    box = spec.worldbody.add_body(name=PAYLOAD_BODY, pos=(.17, 0, .8))
    box.add_freejoint(name=PAYLOAD_JOINT)
    box.mass = mass_kg
    box.inertia = mass_kg * np.array((HALF_SIZE[1]**2 + HALF_SIZE[2]**2,
                                     HALF_SIZE[0]**2 + HALF_SIZE[2]**2,
                                     HALF_SIZE[0]**2 + HALF_SIZE[1]**2)) / 3
    # These are LOCAL inertial coordinates. The body's initial world position
    # must never become an offset centre of mass for the homogeneous box.
    box.ipos = (0., 0., 0.)
    box.iquat = (1., 0., 0., 0.)
    box.explicitinertial = True
    box.add_geom(name=PAYLOAD_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=HALF_SIZE, mass=0, contype=1, conaffinity=1,
                 friction=(.7, .005, .0001), solref=(.02, 1), rgba=(.85, .43, .08, 1))
    for support in SUPPORT_GEOMS:
        spec.add_pair(name=f"{support}_payload", geomname1=support, geomname2=PAYLOAD_GEOM,
                      condim=3, friction=(.7, .7, .005, .0001, .0001),
                      solref=(.0105, 1), solimp=(.9, .95, .001, .5, 2))
    # .0105 is the equal-solmix average of official arm .001 and box .02.
    model = spec.compile()
    validate_model(model, JOINT_NAMES)
    payload_body = model.body(PAYLOAD_BODY).id
    expected_inertia = mass_kg * np.array((HALF_SIZE[1]**2 + HALF_SIZE[2]**2,
                                         HALF_SIZE[0]**2 + HALF_SIZE[2]**2,
                                         HALF_SIZE[0]**2 + HALF_SIZE[1]**2)) / 3
    if (not np.allclose(model.body_ipos[payload_body], 0., rtol=0., atol=1e-12)
            or not np.allclose(model.body_iquat[payload_body], (1., 0., 0., 0.), rtol=0., atol=1e-12)
            or not math.isclose(float(model.body_mass[payload_body]), mass_kg, rel_tol=0., abs_tol=1e-12)
            or not np.allclose(model.body_inertia[payload_body], expected_inertia, rtol=1e-12, atol=1e-12)
            or not np.all(model.body_inertia[payload_body] > 0.)):
        raise ValueError("Payload inertial frame/mass/inertia differs from a centred homogeneous box")
    data = mujoco.MjData(model)
    root = model.joint(ROOT_JOINT).id
    radr = int(model.jnt_qposadr[root])
    data.qpos[radr:radr+7] = (*SIM_INIT_POS, 1, 0, 0, 0)
    for name, q in zip(JOINT_NAMES, DEFAULT_JOINT_POS):
        data.qpos[int(model.jnt_qposadr[model.joint(name).id])] = q
    box_jid = model.joint(PAYLOAD_JOINT).id
    badr = int(model.jnt_qposadr[box_jid])
    data.qpos[badr:badr+7] = (.17, 0, .8, 1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    feet = [model.geom(name).id for name in FOOT_GEOMS]
    # Sole-box extent in world Z gives exact low point for the initial pose.
    sole_min = min(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3,3)[2]) @ model.geom_size[g]
                   for g in feet)
    root_correction = .0001 - sole_min
    data.qpos[radr + 2] += root_correction
    mujoco.mj_forward(model, data)
    payload_geom = model.geom(PAYLOAD_GEOM).id
    support_ids = [model.geom(name).id for name in SUPPORT_GEOMS]

    def clearance(z):
        data.qpos[badr + 2] = z
        mujoco.mj_forward(model, data)
        # Use actual narrowphase contacts for the descending bracket. A long
        # 2 m mesh distance query can return a spurious zero for distant meshes
        # in the tested MuJoCo runtime; it must not be interpreted as support.
        distances = [float(c.dist) for c in data.contact
                     if payload_geom in (c.geom1, c.geom2)
                     and (c.geom2 if c.geom1 == payload_geom else c.geom1) in support_ids]
        return min(distances, default=.02)

    # Find the first forearm collision while descending from above. Only
    # initialization uses geometry; the policy is never given this position.
    previous_z = float(data.qpos[radr+2] + .6)
    if clearance(previous_z) <= 0:
        raise ValueError("Cannot initialize box above forearms")
    low = high = None
    for z in np.linspace(previous_z, data.qpos[radr+2] - .15, 151)[1:]:
        if clearance(float(z)) <= 0:
            low, high = float(z), previous_z
            break
        previous_z = float(z)
    if low is None:
        raise ValueError("Box footprint does not intersect official forearm meshes")
    for _ in range(40):
        mid = (low + high) / 2
        if clearance(mid) <= 0:
            low = mid
        else:
            high = mid
    clearance(high + 1e-6)
    distances = {}
    for gid in range(model.ngeom):
        if gid == payload_geom or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        distance = float(mujoco.mj_geomDistance(model, data, gid, payload_geom, .1, None))
        distances[model.geom(gid).name] = distance
        if distance < -1e-5:
            raise ValueError(f"Initial payload penetrates {model.geom(gid).name} by {-distance} m")
    if any(abs(distances[name]) > .001 for name in SUPPORT_GEOMS):
        raise ValueError(f"Initial level box does not sit above both official forearms: "
                         f"z={float(data.qpos[badr+2])}, gaps="
                         f"{ {name: distances[name] for name in SUPPORT_GEOMS} }")
    after = hashlib.sha256(asset_xml.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("Vendor asset changed during scene construction")
    audit = {
        "source_asset": str(asset_xml), "source_xml_sha256": before,
        "vendor_xml_unchanged": True,
        "overlay": "named original collision geoms; added independent free box and explicit forearm/box friction pairs",
        "robot_geometry_unchanged": True,
        "payload_body_name": PAYLOAD_BODY, "payload_geom_name": PAYLOAD_GEOM,
        "payload_joint_name": PAYLOAD_JOINT, "robot_root_joint_name": ROOT_JOINT,
        "support_geom_names": list(SUPPORT_GEOMS), "allowed_auxiliary_geom_names": [],
        "foot_geom_names": list(FOOT_GEOMS), "ground_geom_name": "ground",
        "payload_mass_kg": mass_kg, "payload_full_size_m": (2*HALF_SIZE).tolist(),
        "payload_diagonal_inertia_kg_m2": model.body_inertia[model.body(PAYLOAD_BODY).id].tolist(),
        "payload_local_center_of_mass_m": model.body_ipos[payload_body].tolist(),
        "payload_local_inertial_orientation_wxyz": model.body_iquat[payload_body].tolist(),
        "initial_height_reference_m": float(data.qpos[badr+2]),
        "initial_payload_center_m": data.qpos[badr:badr+3].tolist(),
        "initial_foot_clearance_m": .0001, "initial_root_height_correction_m": float(root_correction),
        "initial_payload_collision_clearances_m": distances,
        "initial_collision_clearance_query_upper_bound_m": .1,
        "box_forearm_effective_sliding_friction": .7,
        "box_forearm_pair_condim": 3, "box_forearm_pair_solref": [.0105, 1.],
        "foot_ground_effective_sliding_friction": 1.,
        "attachment_constraints": int(model.neq), "robot_plus_box_free_joints": 2,
        "robot_mass_kg": float(model.body_mass.sum()-mass_kg),
        "differences_from_training": [
            "Official mesh forearms remain, versus training cylinder forearms and end balls; no end-ball auxiliary support here.",
            "Official box foot colliders remain, versus training meshes.",
            "Official contacts use their original solref .001; only forearm/payload explicit pairs have .0105.",
            "Official explicit PD torque motors / Euler differ from training implicit position actuators / implicitfast.",
            "Official robot mass 19.666 kg versus training 19.7968997 kg; XML motor limits remain unchanged.",
        ],
    }
    return model, data.qpos.copy(), audit
