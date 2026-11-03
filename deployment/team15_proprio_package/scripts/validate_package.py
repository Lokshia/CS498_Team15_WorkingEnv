"""Audit exported proprioceptive actors and run loaded-box deployment simulation.

Uses ordinary MuJoCo and the Team15 extension of booster_deploy's controller;
no mjlab, camera, ROS, or robot commands. Box state is read only by this scorer.
Every rollout is one original episode; no resets or replacement successes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

DEPLOY_COMMIT = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"
ASSETS_COMMIT = "12a97516f57469078707afb587a5f84c1cb110d1"
TASK_NAME = "team15_proprio_hold_sim"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def verify_checkout(root, expected_revision):
    """Allow the added untracked task, never modified tracked upstream code/assets."""
    actual = revision(root)
    if actual != expected_revision:
        raise ValueError(f"Upstream revision mismatch at {root}: {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError(f"Tracked upstream files are modified at {root}:\n{dirty}")
    return dict(revision=actual, tracked_files_clean=True)


def source_hashes(deploy, asset_xml):
    """Bind evidence to the exact installed overlay, actors, asset XML and scorer."""
    task_root = deploy / "tasks/team15_proprio"
    code = sorted(task_root.rglob("*.py"))
    if not code:
        raise ValueError("No installed Team15 task source to fingerprint")
    models = task_root / "models"
    required = [models / f"proprio_hold.{suffix}" for suffix in
                ("onnx", "torchscript", "contract.json", "test_vectors.npz")]
    if not all(p.is_file() for p in required):
        raise ValueError("Missing model/contract/test-vector files to fingerprint")
    files = {"validator": Path(__file__), "official_robot_xml": Path(asset_xml)}
    for path in code + sorted(p for p in models.iterdir() if p.is_file()):
        files[path.relative_to(deploy).as_posix()] = path
    return {name: dict(path=str(path.resolve(strict=True)), sha256=sha(path))
            for name, path in files.items()}


def export_tests(model_dir):
    import onnxruntime as ort
    from tasks.team15_proprio.policy import load_export_contract
    name = "proprio_hold"
    contract = load_export_contract(model_dir / f"{name}.onnx")
    vector_path = model_dir / f"{name}.test_vectors.npz"
    if sha(vector_path) != contract["test_vectors_sha256"]:
        raise ValueError("Test-vector SHA-256 mismatch")
    with np.load(vector_path, allow_pickle=False) as vectors:
        observations, expected = vectors["observations"], vectors["expected_actions"]
    if observations.ndim != 2 or observations.shape[1] != 72 or expected.shape != (len(observations), 22):
        raise ValueError("Invalid proprioceptive test-vector shapes")
    if not np.isfinite(observations).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite export test vectors")
    rows = []
    for ext in ("onnx", "torchscript"):
        path = model_dir / f"{name}.{ext}"
        if sha(path) != contract[f"{ext}_sha256"]:
            raise ValueError(f"{ext} model SHA-256 mismatch")
        if ext == "onnx":
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
                raise ValueError("Expected one ONNX input and output")
            if session.get_inputs()[0].shape != [1, 72] or session.get_outputs()[0].shape != [1, 22]:
                raise ValueError("Unexpected ONNX shape")
            infer = lambda x: session.run(None, {session.get_inputs()[0].name: x})[0]
        else:
            module = torch.jit.load(str(path), map_location="cpu").eval()
            def infer(x):
                with torch.inference_mode():
                    return module(torch.from_numpy(x)).numpy()
        actual = np.concatenate([infer(x[None].astype(np.float32)) for x in observations])
        good_shape = actual.shape == expected.shape
        passed = bool(good_shape and np.isfinite(actual).all() and np.max(np.abs(actual)) <= 1.000001
                      and np.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        rows.append(dict(model=name, format=ext, samples=len(observations), passed=passed,
                         max_absolute_error=float(np.max(np.abs(actual-expected))) if good_shape else None,
                         atol=1e-5, rtol=1e-5, sha256=sha(path)))
    return rows


def model_audit(controller):
    import mujoco
    from tasks.team15_proprio.contract import JOINT_NAMES
    m, d, cfg = controller.mj_model, controller.mj_data, controller.cfg
    joints = np.array([m.joint(name).id for name in JOINT_NAMES], dtype=int)
    if len(set(joints.tolist())) != 22 or (m.nq, m.nv, m.nu) != (36, 34, 22):
        raise ValueError("Expected 22 actuated hinges plus free robot and free box")
    if m.neq != 0 or (m.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum() != 2:
        raise ValueError("Box or robot is constrained")
    if not np.array_equal(m.actuator_trnid[:, 0], joints):
        raise ValueError("Actuator joint mapping differs from controller ordering")
    if not np.allclose(m.actuator_gear[:, 0], 1) or not np.allclose(m.actuator_gear[:, 1:], 0):
        raise ValueError("Unexpected actuator gearing")
    payload_body = m.body("team15_payload").id
    payload_geom = m.geom("team15_payload_box").id
    payload_joint = m.joint("team15_payload_free").id
    mass = float(m.body_mass[payload_body])
    half_size = m.geom_size[payload_geom]
    expected_inertia = mass*np.array((half_size[1]**2+half_size[2]**2,
                                     half_size[0]**2+half_size[2]**2,
                                     half_size[0]**2+half_size[1]**2))/3
    if (m.geom_type[payload_geom] != mujoco.mjtGeom.mjGEOM_BOX
            or not np.isfinite(mass) or mass <= 0 or not np.all(half_size > 0)
            or not np.allclose(m.geom_pos[payload_geom], 0., rtol=0., atol=1e-12)
            or not np.allclose(m.geom_quat[payload_geom], [1., 0., 0., 0.], rtol=0., atol=1e-12)
            or not np.allclose(m.body_ipos[payload_body], 0., rtol=0., atol=1e-12)
            or not np.allclose(m.body_iquat[payload_body], [1., 0., 0., 0.], rtol=0., atol=1e-12)
            or not np.all(m.body_inertia[payload_body] > 0)
            or not np.allclose(m.body_inertia[payload_body], expected_inertia, rtol=1e-12, atol=1e-12)):
        raise ValueError("Payload must have the centred inertial frame and inertia of the homogeneous box")
    if m.jnt_type[payload_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Payload is not a free body")
    base_joint = next(i for i in range(m.njnt)
                      if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and i != payload_joint)
    ground = m.geom("ground").id
    feet = []
    for side in ("left", "right"):
        body = m.body(f"{side}_ankle_roll_link").id
        geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body
                 and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
        if len(geoms) != 1:
            raise ValueError("Expected one sole collision box per foot")
        feet.append(geoms[0])
    scene_audit = controller.scene_audit
    support_names = scene_audit["support_geom_names"]
    if len(support_names) != 2:
        raise ValueError("Two named forearm support surfaces are required")
    support = [m.geom(name).id for name in support_names]
    auxiliary = [m.geom(name).id for name in scene_audit.get("allowed_auxiliary_geom_names", [])]
    layout = dict(joints=joints, qadr=m.jnt_qposadr[joints].copy(), vadr=m.jnt_dofadr[joints].copy(),
                  ground=ground, feet=feet, support=support, auxiliary=auxiliary,
                  payload_body=payload_body, payload_geom=payload_geom,
                  payload_qadr=int(m.jnt_qposadr[payload_joint]), payload_vadr=int(m.jnt_dofadr[payload_joint]),
                  base_qadr=int(m.jnt_qposadr[base_joint]), base_vadr=int(m.jnt_dofadr[base_joint]),
                  trunk=m.body("trunk").id)
    ranges = m.jnt_range[joints]
    if not m.jnt_limited[joints].all() or not np.isfinite(ranges).all():
        raise ValueError("Joint limit audit requires 22 finite limited joints")
    report = dict(nq=m.nq, nv=m.nv, nu=m.nu, equality_constraints=m.neq,
                  joint_names=list(JOINT_NAMES), robot_mass_kg=float(m.body_mass.sum()-m.body_mass[payload_body]),
                  payload_mass_kg=float(m.body_mass[payload_body]), payload_half_size_m=m.geom_size[payload_geom].tolist(),
                  payload_inertia_kg_m2=m.body_inertia[payload_body].tolist(),
                  payload_local_center_of_mass_m=m.body_ipos[payload_body].tolist(),
                  payload_local_inertial_orientation_wxyz=m.body_iquat[payload_body].tolist(),
                  payload_expected_uniform_box_inertia_kg_m2=expected_inertia.tolist(),
                  payload_qpos_address=layout["payload_qadr"], joint_qpos_addresses=layout["qadr"].tolist(),
                  joint_dof_addresses=layout["vadr"].tolist(), joint_ranges_rad=ranges.tolist(),
                  actuator_joint_ids=m.actuator_trnid[:, 0].tolist(),
                  xml_actuator_force_limits=m.actuator_forcerange.tolist(),
                  configured_effort_limits=list(cfg.robot.effort_limit),
                  policy_dt_s=cfg.policy_dt, physics_dt_s=float(m.opt.timestep), decimation=cfg.mujoco.decimation,
                  simulation_kp=list(cfg.robot.joint_stiffness), simulation_kd=list(cfg.robot.joint_damping),
                  integrator=int(m.opt.integrator), impratio=float(m.opt.impratio),
                  ground_friction=m.geom_friction[ground].tolist(), sole_friction=m.geom_friction[feet].tolist(),
                  payload_friction=m.geom_friction[payload_geom].tolist(),
                  support_friction=m.geom_friction[support].tolist(), scene=scene_audit,
                  note="Official robot asset plus documented payload/contact overlay and 22-joint index adapter; not an unchanged upstream controller.")
    return report, layout


def contact_sample(model, data, layout):
    """World forces on feet/box; never supplies observations to the actor."""
    import mujoco
    feet, support, box, ground = layout["feet"], layout["support"], layout["payload_geom"], layout["ground"]
    foot_force = np.zeros((2, 3)); foot_contact = np.zeros(2, bool)
    support_force = np.zeros((2, 3)); support_contact = np.zeros(2, bool)
    forbidden = False; loaded_forbidden = False
    allowed = set(support) | set(layout["auxiliary"])
    for index in range(data.ncon):
        c = data.contact[index]
        pair = (int(c.geom1), int(c.geom2))
        if box not in pair and ground not in pair:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)
        world = c.frame.reshape(3, 3).T @ local[:3]
        if ground in pair:
            for i, foot in enumerate(feet):
                if foot in pair:
                    foot_contact[i] = True
                    foot_force[i] += world if c.geom2 == foot else -world
        if box in pair:
            other = pair[1] if pair[0] == box else pair[0]
            force_on_box = world if c.geom2 == box else -world
            if other in support:
                i = support.index(other)
                support_contact[i] = True
                support_force[i] += force_on_box
            elif other not in allowed:
                forbidden = True
                loaded_forbidden |= bool(np.dot(force_on_box, force_on_box) > 1e-12)
    return dict(foot_force_n=foot_force, foot_contact=foot_contact,
                forearm_force_on_payload_n=support_force, forearm_contact=support_contact,
                payload_forearm_supported=bool((support_contact & (support_force[:, 2] > 1e-6)).all()),
                forbidden_contact=forbidden, loaded_forbidden_contact=loaded_forbidden)


def sample(controller, layout, origin_xy, reference_height, support_gap, forbidden_substep):
    m, d = controller.mj_model, controller.mj_data
    contact = contact_sample(m, d, layout)
    support_gap = 0 if contact["payload_forearm_supported"] else support_gap + 1
    box = layout["payload_body"]
    angle = lambda value: math.degrees(math.acos(float(np.clip(value, -1, 1))))
    sole_tilts = [angle(d.geom_xmat[g].reshape(3, 3)[2, 2]) for g in layout["feet"]]
    trunk_tilt = angle(d.xmat[layout["trunk"]].reshape(3, 3)[2, 2])
    payload_tilt = angle(d.xmat[box].reshape(3, 3)[2, 2])
    payload_height = float(d.xpos[box, 2])
    height_error = payload_height-reference_height
    base_qadr = layout["base_qadr"]
    base_height = float(d.qpos[base_qadr+2])
    forbidden = bool(contact["forbidden_contact"] or forbidden_substep)
    fell = bool(base_height < .28 or trunk_tilt > 60 or payload_height < .20)
    valid = bool(payload_tilt < 10 and abs(height_error) < .05 and not fell
                 and not forbidden and support_gap <= 6)
    ranges = m.jnt_range[layout["joints"]]
    qpos, qvel = d.qpos[layout["qadr"]].copy(), d.qvel[layout["vadr"]].copy()
    excess = np.maximum(np.maximum(ranges[:, 0]-qpos, qpos-ranges[:, 1]), 0.)
    row = dict(time_s=float(d.time), foot_tilt_deg=sole_tilts,
               foot_force_n=contact["foot_force_n"], foot_contact=contact["foot_contact"],
               trunk_tilt_deg=trunk_tilt, base_height_m=base_height,
               drift_m=float(np.linalg.norm(d.qpos[base_qadr:base_qadr+2]-origin_xy)),
               joint_position=qpos, joint_velocity=qvel, actuator_torque=d.qfrc_actuator[layout["vadr"]].copy(),
               actual_joint_limit_excess_rad=excess,
               payload_position_w=d.xpos[box].copy(), payload_height_m=payload_height,
               payload_tilt_deg=payload_tilt, height_error_m=height_error,
               forearm_contact=contact["forearm_contact"],
               forearm_force_on_payload_n=contact["forearm_force_on_payload_n"],
               payload_forearm_supported=contact["payload_forearm_supported"],
               payload_disallowed_contact=forbidden, support_gap_steps=support_gap,
               support_lost=support_gap > 6, fell=fell, valid_hold=valid)
    return row, support_gap


def score(rows, steps, failure, nominal):
    """A complete original episode, including all startup frames, must hold."""
    steady = [r for r in rows if r["time_s"] >= 2.0]
    complete = bool(failure is None and len(rows) == steps)
    hold = bool(complete and all(r["valid_hold"] for r in rows))
    value = lambda key: np.asarray([r[key] for r in steady])
    metrics = {}
    if steady:
        arm_error = value("joint_position")[:, 2:10]-np.asarray(nominal)[2:10]
        flat = value("foot_contact").all(1) & (value("foot_tilt_deg") < 5).all(1)
        metrics = dict(steady_samples=len(steady), flat_contact_fraction=float(flat.mean()),
                       foot_tilt_mean_deg=value("foot_tilt_deg").mean(0).tolist(),
                       foot_tilt_max_deg=value("foot_tilt_deg").max(0).tolist(),
                       foot_vertical_force_mean_n=value("foot_force_n")[:, :, 2].mean(0).tolist(),
                       trunk_tilt_p95_deg=float(np.percentile(value("trunk_tilt_deg"), 95)),
                       arm_error_rms_rad=float(np.sqrt(np.mean(arm_error**2))),
                       joint_velocity_rms_rad_s=float(np.sqrt(np.mean(value("joint_velocity")**2))),
                       joint_torque_rms_nm=float(np.sqrt(np.mean(value("actuator_torque")**2))),
                       payload_supported_fraction=float(value("payload_forearm_supported").mean()),
                       payload_tilt_max_deg=float(value("payload_tilt_deg").max()),
                       payload_height_error_max_m=float(np.abs(value("height_error_m")).max()))
    drift = max((r["drift_m"] for r in rows), default=0.)
    strict = bool(hold and steady and metrics["flat_contact_fraction"] >= .95
                  and metrics["trunk_tilt_p95_deg"] < 5 and metrics["arm_error_rms_rad"] < .15
                  and metrics["joint_velocity_rms_rad_s"] < .2 and drift < .1)
    return dict(full_horizon_survival=complete, hold_success=hold, strict_standing_pass=strict,
                max_drift_m=float(drift),
                all_step_max_payload_tilt_deg=max((r["payload_tilt_deg"] for r in rows), default=None),
                all_step_max_abs_height_error_m=max((abs(r["height_error_m"]) for r in rows), default=None),
                all_step_max_support_gap=max((r["support_gap_steps"] for r in rows), default=None),
                forbidden_contact_seen=any(r["payload_disallowed_contact"] for r in rows), **metrics)


def rollout(cfg_factory, seconds, seed, mass, randomized):
    import mujoco
    from tasks.team15_proprio.simulator import ProprioHoldMujocoController
    cfg = cfg_factory()
    controller = ProprioHoldMujocoController(cfg, payload_mass_kg=mass)
    audit, layout = model_audit(controller)
    m, d = controller.mj_model, controller.mj_data
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-.05, .05, 2) if randomized else np.zeros(2)
    yaw = float(rng.uniform(-.05, .05)) if randomized else 0.
    velocity = rng.uniform(-.02, .02, 2) if randomized else np.zeros(2)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    quaternion = np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
    for qadr, vadr in ((layout["base_qadr"], layout["base_vadr"]),
                       (layout["payload_qadr"], layout["payload_vadr"])):
        d.qpos[qadr:qadr+2] = rotation @ d.qpos[qadr:qadr+2] + xy
        current = d.qpos[qadr+3:qadr+7].copy()
        mujoco.mju_mulQuat(d.qpos[qadr+3:qadr+7], quaternion, current)
        d.qvel[vadr:vadr+2] = velocity
    mujoco.mj_forward(m, d)
    initial_qpos, initial_qvel = d.qpos.copy(), d.qvel.copy()
    reference_height = float(d.xpos[layout["payload_body"], 2])
    origin = d.qpos[layout["base_qadr"]:layout["base_qadr"]+2].copy()
    controller.update_state()
    controller.start()
    substep_flags = []
    controller.validation_substep_callback = lambda: substep_flags.append(
        contact_sample(m, d, layout)["loaded_forbidden_contact"])
    rows, failure, gap = [], None, 0
    counts = np.zeros(22, dtype=np.int64); maximum = np.zeros(22)
    steps = round(seconds/cfg.policy_dt)
    if not math.isclose(steps*cfg.policy_dt, seconds):
        raise ValueError("Duration must be a multiple of 0.02 seconds")
    try:
        for _ in range(steps):
            controller.update_state()
            obs = controller.policy.compute_observation().detach().cpu().numpy().copy()
            target = controller.policy_step()
            if not controller.is_running:
                failure = "policy_stopped"
                break
            if target.shape != (22,) or not torch.isfinite(target).all():
                failure = "invalid_joint_targets"
                break
            target_np = target.detach().cpu().numpy().copy()
            ranges = m.jnt_range[layout["joints"]]
            excess = np.maximum(np.maximum(ranges[:, 0]-target_np, target_np-ranges[:, 1]), 0.)
            counts += excess > 1e-6; maximum = np.maximum(maximum, excess)
            raw_action = controller.policy.last_raw_action.detach().cpu().numpy().copy()
            effective = controller.policy.last_action.detach().cpu().numpy().copy()
            substep_flags.clear()
            controller.ctrl_step(target)
            if len(substep_flags) != cfg.mujoco.decimation:
                raise RuntimeError("Per-physics-substep contact callback did not run")
            if not all(np.isfinite(v).all() for v in (d.qpos, d.qvel, d.qfrc_actuator)):
                failure = "nonfinite_state"
                break
            row, gap = sample(controller, layout, origin, reference_height, gap, any(substep_flags))
            row.update(actor_observation=obs, raw_action=raw_action, effective_action=effective,
                       sent_joint_target=target_np, forbidden_contact_substeps=np.asarray(substep_flags, bool))
            rows.append(row)
            if row["fell"] or row["support_lost"] or row["payload_disallowed_contact"]:
                failure = "fall" if row["fell"] else "support_lost" if row["support_lost"] else "forbidden_box_contact"
                break
    except (ValueError, RuntimeError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        controller.validation_substep_callback = None
        controller.stop()
    summary = dict(seed=seed, payload_mass_kg=mass, seconds_requested=seconds,
                   steps_expected=steps, steps_completed=len(rows), failure=failure,
                   initial_qpos=initial_qpos.tolist(), initial_qvel=initial_qvel.tolist(),
                   initial_common_xy_m=xy.tolist(), initial_common_yaw_rad=yaw,
                   initial_common_xy_velocity_m_s=velocity.tolist(), payload_reference_height_m=reference_height,
                   commanded_targets_within_xml_limits=not bool(counts.any()),
                   target_limit_exceedance_steps_per_joint=counts.tolist(),
                   target_limit_max_exceedance_rad_per_joint=maximum.tolist(),
                   projection=controller.policy.projection_statistics(),
                   **score(rows, steps, failure, cfg.robot.default_joint_pos))
    summary["actual_joint_limit_max_excess_rad"] = (
        np.max(np.asarray([r["actual_joint_limit_excess_rad"] for r in rows]), axis=0).tolist() if rows else None)
    return summary, rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--mass-kg", type=float, default=.5)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=10011)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or not math.isfinite(args.seconds) or args.seconds <= 2:
        parser.error("Use positive episodes and finite duration greater than 2 seconds")
    if not math.isfinite(args.mass_kg) or args.mass_kg < .001:
        parser.error("A physical box needs a finite mass >= 0.001 kg")
    deploy, assets = args.deploy_root.resolve(strict=True), args.assets_root.resolve(strict=True)
    upstream_before = dict(deploy=verify_checkout(deploy, DEPLOY_COMMIT),
                           assets=verify_checkout(assets, ASSETS_COMMIT))
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(assets / "src")); sys.path.insert(0, str(deploy))
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tasks.team15_proprio  # noqa: F401
    import booster_assets
    from booster_deploy.utils.registry import get_task
    asset_xml = (Path(booster_assets.BOOSTER_ASSETS_DIR)/"robots/K1/K1_22dof.xml").resolve(strict=True)
    asset_xml.relative_to(assets)  # Do not silently audit a different installed asset distribution.
    task_file = Path(tasks.team15_proprio.__file__).resolve(strict=True)
    task_file.relative_to(deploy / "tasks/team15_proprio")
    before = source_hashes(deploy, asset_xml)
    factory = lambda: deepcopy(get_task(TASK_NAME))
    report = dict(schema="team15-proprio-deploy-validation-v1", hardware_ready=False,
                  deploy_commit=DEPLOY_COMMIT, assets_commit=ASSETS_COMMIT,
                  source_hashes=before, upstream_before=upstream_before,
                  platform=platform.platform(), python=sys.version,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "onnxruntime", "mujoco", "numpy")},
                  exports=export_tests(deploy / "tasks/team15_proprio/models"), episodes=[],
                  actor_inputs="72 IMU/joint/previous-action features only; no box state, mass or camera",
                  limitations=["No hardware or ROS control was run.",
                               "Official assets with a documented payload/contact overlay and custom indexed simulator adapter.",
                               "Arms raised, box initially supported; no arm-raising or load/unload transitions.",
                               "Simulation gains are not reviewed real motor-side values."])
    if not args.export_only:
        for i in range(args.episodes):
            summary, rows, audit = rollout(factory, args.seconds, args.seed+i, args.mass_kg, i != 0)
            report["episodes"].append(summary); report["model_audit"] = audit
            if rows:
                np.savez_compressed(args.output / f"episode_{i:02d}.npz",
                                    **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    report["export_checks_passed"] = all(r["passed"] for r in report["exports"])
    report["full_horizon_survivals"] = sum(e["full_horizon_survival"] for e in report["episodes"])
    report["hold_successes"] = sum(e["hold_success"] for e in report["episodes"])
    report["strict_standing_passes"] = sum(e["strict_standing_pass"] for e in report["episodes"])
    report["commanded_targets_within_xml_limits"] = all(e["commanded_targets_within_xml_limits"] for e in report["episodes"])
    try:
        report["source_hashes_after"] = source_hashes(deploy, asset_xml)
        report["upstream_after"] = dict(deploy=verify_checkout(deploy, DEPLOY_COMMIT),
                                        assets=verify_checkout(assets, ASSETS_COMMIT))
        if report["source_hashes_after"] != before:
            raise ValueError("Source/model/asset/validator files changed during validation")
        report["source_integrity_passed"] = True
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        report["source_integrity_passed"] = False
        report["source_integrity_error"] = f"{type(exc).__name__}: {exc}"
    report["all_passed"] = bool(report["source_integrity_passed"] and report["export_checks_passed"] and (args.export_only or
                              (report["strict_standing_passes"] == args.episodes and report["commanded_targets_within_xml_limits"])))
    report["criteria"] = ("Entire original episode: box tilt<10deg, height error<.05m, "
        "base height>=.28m/trunk tilt<=60deg/box height>=.20m; no forbidden box contacts; "
        "both forearms upward force>1e-6N with at most 6 consecutive missing-support control samples. "
        "Steady t>=2s: >=95% both feet contact and both sole tilts<5deg, trunk p95<5deg, "
        "arm pose RMS<.15rad, joint velocity RMS<.2rad/s; whole-episode drift<.10m. "
        "Sent targets within official XML ranges, tolerance1e-6rad. No reset replacement.")
    report["contact_sampling"] = "Final physics-substep forces and geometric contacts; forbidden loaded contacts additionally sampled at every substep"
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Report: {path}; all_passed={report['all_passed']}")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
