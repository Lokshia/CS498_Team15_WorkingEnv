"""Replay one trusted checkpoint without resets and measure physical foot support.

This diagnostic does not change the environment, rewards, or policy. It stores
the *last physics substep* of each control step, not a substep average. MuJoCo
contact sensor netforce is the primary-on-secondary wrench; for foot-primary,
terrain-secondary sensors its negative is the ground reaction on that foot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from capture_joint_trace import make_demo_args, _positive, _nonnegative


def sole_tilt_degrees(quaternion_wxyz):
    """Angle between the sole site's local +Z and world +Z (yaw invariant)."""
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternions")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero quaternion")
    q = q / norms
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1)))


def summarize_window(data, select):
    if not np.any(select):
        return None
    fz = data["ground_reaction_world_n"][select, :, 2]
    contact = data["foot_contact_count"][select] > 0
    tilt = data["sole_tilt_deg"][select]
    total = fz.sum(axis=-1)
    loaded = total > 1.0
    ratio = np.clip(fz[loaded], 0, None)
    ratio = ratio[:, 0] / ratio.sum(axis=-1).clip(1e-9)
    torque = data["joint_torque_nm"][select]
    return {
        "frames": int(np.sum(select)),
        "time_first_s": float(data["time_s"][select][0]),
        "time_last_s": float(data["time_s"][select][-1]),
        "vertical_force_mean_n_left_right": fz.mean(axis=0).tolist(),
        "vertical_force_min_n_left_right": fz.min(axis=0).tolist(),
        "vertical_force_max_n_left_right": fz.max(axis=0).tolist(),
        "total_vertical_force_mean_n": float(total.mean()),
        "loaded_frames": int(np.sum(loaded)),
        "left_load_fraction_mean_when_loaded": float(ratio.mean()) if len(ratio) else None,
        "left_load_fraction_p05_p95_when_loaded": np.quantile(ratio, [.05, .95]).tolist() if len(ratio) else None,
        "fraction_with_each_foot_at_least_20percent_when_loaded": float(((ratio >= .2) & (ratio <= .8)).mean()) if len(ratio) else None,
        "sole_tilt_mean_deg_left_right": tilt.mean(axis=0).tolist(),
        "sole_tilt_p95_deg_left_right": np.quantile(tilt, .95, axis=0).tolist(),
        "sole_tilt_max_deg_left_right": tilt.max(axis=0).tolist(),
        "both_feet_contact_fraction": float(contact.all(axis=-1).mean()),
        "both_feet_contact_and_tilt_under_5deg_fraction": float((contact.all(axis=-1) & (tilt < 5).all(axis=-1)).mean()),
        "both_feet_contact_and_tilt_under_10deg_fraction": float((contact.all(axis=-1) & (tilt < 10).all(axis=-1)).mean()),
        "robot_tilt_mean_deg": float(data["robot_tilt_deg"][select].mean()),
        "robot_tilt_max_deg": float(data["robot_tilt_deg"][select].max()),
        "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(data["joint_vel_rad_s"][select] ** 2))),
        "joint_torque_rms_nm": float(np.sqrt(np.mean(torque ** 2))),
        "joint_torque_rms_nm_by_joint": np.sqrt(np.mean(torque ** 2, axis=0)).tolist(),
        "joint_torque_max_abs_nm_by_joint": np.abs(torque).max(axis=0).tolist(),
    }


def summarize_leg_mirror(data, select):
    """Independent named-joint sagittal mirror errors, in degrees, not a reward."""
    if not np.any(select):
        return None
    joint_names = np.asarray(data["joint_names"]).astype(str).tolist()
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("Duplicate joint names")
    pairs = ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
    left = [joint_names.index(f"Left_{name}") for name in pairs]
    right = [joint_names.index(f"Right_{name}") for name in pairs]
    positions = np.asarray(data["joint_pos_rad"])[select]
    if positions.ndim != 2 or positions.shape[1] != len(joint_names) or not np.isfinite(positions).all():
        raise ValueError("Finite joint positions with named column mapping required")
    signs = np.asarray([1, -1, -1, 1, 1, -1])
    errors = np.degrees(positions[:, left] - positions[:, right] * signs)
    return {
        "pair_names": list(pairs), "right_joint_mirror_sign": signs.tolist(),
        "convention": "q_left - sign*q_right; named joints; degrees. Zero is an exact mirrored leg pose.",
        "rms_deg_over_frames_and_pairs": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_deg_over_frames_and_pairs": float(np.abs(errors).max()),
        "rms_deg_by_pair": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        "mean_signed_deg_by_pair": errors.mean(axis=0).tolist(),
        "max_abs_deg_by_pair": np.abs(errors).max(axis=0).tolist(),
        "all_pairs_within_5deg_fraction": float((np.abs(errors) <= 5).all(axis=1).mean()),
        "limitations": "A mirrored joint pose is a geometric preference, not proof of balanced forces or stability.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mass", type=_nonnegative, default=.4949224923474894)
    parser.add_argument("--seconds", type=_positive, default=20.)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--mass-blind", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from lifting_lab import cli, mdp
    import torch

    runtime_at_start = cli.runtime_info(args.device)
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    demo_args = make_demo_args(args, cli)
    cli._validate_checkpoint_support(demo_args)
    args.output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env, mass = cli._env(demo_args, seconds=args.seconds + 1, auto_reset=False, num_envs=1)
        policy = cli._policy(env, demo_args)
        obs, _ = env.reset(seed=args.seed)
        robot = env.scene["robot"]
        sensor = env.scene["feet_ground_contact"]
        if sensor.cfg.reduce != "netforce" or sensor.cfg.num_slots != 1:
            raise ValueError("Expected one world-frame netforce slot per foot subtree")
        primary_names = sensor.primary_names
        order = []
        for foot in ("left_foot_link", "right_foot_link"):
            matches = [i for i, name in enumerate(primary_names) if name.split("/")[-1] == foot]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous foot mapping {foot}: {primary_names}")
            order.append(matches[0])
        sole_ids = robot.find_sites(("left_foot_sole", "right_foot_sole"), preserve_order=True)[0]
        arrays = {key: [] for key in ("time_s", "sensor_force_world_n", "ground_reaction_world_n",
            "foot_contact_count", "sole_quaternion_wxyz", "sole_position_world_m", "sole_tilt_deg",
            "joint_torque_nm", "joint_pos_rad", "joint_vel_rad_s", "robot_tilt_deg", "combined_com_world_m")}
        trace = []
        terminated = truncated = False
        dt = float(env.step_dt)
        planned = round(args.seconds / dt)

        def snapshot(tensor):
            value = tensor.detach().cpu().numpy()[0].copy()
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite diagnostic sample")
            return value

        with torch.inference_mode():
            for step in range(planned):
                obs, reward, term, trunc, _ = env.step(policy(obs))
                if not bool(torch.isfinite(reward).all()):
                    raise ValueError("Nonfinite reward")
                force = snapshot(sensor.data.force) [order]
                if force.shape != (2, 3):
                    raise ValueError(f"Unexpected foot force shape {force.shape}")
                quat = snapshot(robot.data.site_quat_w[:, sole_ids])
                metrics = {key: float(value[0]) for key, value in cli._metrics(env).items()}
                sample = {"time_s": (step + 1) * dt,
                    "sensor_force_world_n": force,
                    "ground_reaction_world_n": -force,
                    "foot_contact_count": snapshot(sensor.data.found)[order].reshape(2),
                    "sole_quaternion_wxyz": quat,
                    "sole_position_world_m": snapshot(robot.data.site_pos_w[:, sole_ids]),
                    "sole_tilt_deg": sole_tilt_degrees(quat),
                    "joint_torque_nm": snapshot(robot.data.qfrc_actuator),
                    "joint_pos_rad": snapshot(robot.data.joint_pos),
                    "joint_vel_rad_s": snapshot(robot.data.joint_vel),
                    "robot_tilt_deg": metrics["robot_tilt_deg"],
                    "combined_com_world_m": snapshot(mdp.combined_com_w(env))}
                for key in arrays:
                    arrays[key].append(sample[key])
                trace.append(metrics)
                terminated, truncated = bool(term.any()), bool(trunc.any())
                if terminated or truncated:
                    break
        data = {key: np.asarray(value) for key, value in arrays.items()}
        data["joint_names"] = np.asarray(robot.joint_names, dtype=str)
        total_weight = (mass["robot_mass_kg"] + mass["actual_payload_mass_kg"]) * 9.81
        steady = data["time_s"] >= 2.0
        runtime_at_end = cli.runtime_info(env.device)
        summary = {
            "schema_version": 1, "formal_evaluation": False,
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": cli.file_sha256(args.checkpoint),
            "runtime": runtime_at_start,
            "runtime_source_sha256_at_end": runtime_at_end["source_sha256"],
            "source_files_changed_during_capture": runtime_at_start["source_sha256"] != runtime_at_end["source_sha256"],
            "seed": args.seed, "mass": mass,
            "reward_profile": env.cfg.reward_profile, "requested_seconds": args.seconds,
            "recorded_seconds": float(data["time_s"][-1]), "auto_reset": False,
            "terminated": terminated, "truncated": truncated, "stopped_early": len(trace) < planned,
            "foot_order": ["left", "right"], "sensor_primary_names": primary_names,
            "joint_names": robot.joint_names,
            "reference_total_weight_n": total_weight,
            "sampling": "Post-control-step final substep, 50 Hz; not decimation-averaged force.",
            "force_convention": "netforce sensor sums all contacts in each foot subtree in WORLD frame. Sensor is primary foot on secondary terrain; negative yields ground reaction on foot. No abs or sign guessing.",
            "tilt_convention": "Angle of sole site local +Z to world +Z, from normalized wxyz quaternion; no yaw penalty.",
            "all": summarize_window(data, np.ones(len(trace), dtype=bool)),
            "startup_before_2s": summarize_window(data, ~steady),
            "steady_from_2s": summarize_window(data, steady),
            "task_metrics_final": trace[-1],
            "script_sha256": script_sha256,
            "limitations": ["One replay is not a success-rate estimate or real-robot validation.",
                "Foot vertical forces are simulated ground reactions; joint torques are simulated actuator torques.",
                "Sole tilt and contact presence do not measure contact area, COP, or a true support polygon.",
                "Control-step samples can miss physics-substep peaks."]}
        for key, selected in (("all", np.ones(len(trace), dtype=bool)),
                              ("startup_before_2s", ~steady), ("steady_from_2s", steady)):
            if summary[key] is not None:
                summary[key]["leg_mirror"] = summarize_leg_mirror(data, selected)
        json.dumps(summary, allow_nan=False)
        with (args.output / "stance_trace.npz").open("xb") as handle:
            np.savez_compressed(handle, **data)
        for name, value in (("stance_summary.json", summary), ("task_trace.json", trace)):
            with (args.output / name).open("x", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"output": str(args.output.resolve()), "recorded_seconds": summary["recorded_seconds"],
            "terminated": terminated, "steady_from_2s": summary["steady_from_2s"]}))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
