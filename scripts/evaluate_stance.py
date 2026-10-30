"""Matched batched hold/stance evaluation of trusted local checkpoints.

Each slot contributes one episode only. It is frozen after completion or failure;
resetting its simulator slot never contributes a replacement episode. Original
hold success and an additional strict flat-stance criterion are reported apart.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from capture_joint_trace import make_demo_args, _positive, _nonnegative
from diagnose_stance import sole_tilt_degrees, summarize_window


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected an integer >= 1")
    return value


def strict_stance_success(hold_success, steady_summary, steady_frames, required_steady_frames):
    """Do not promote short surviving fragments to complete stance success."""
    return bool(hold_success and required_steady_frames > 0
                and steady_frames == required_steady_frames and steady_summary is not None
                and steady_summary["both_feet_contact_and_tilt_under_5deg_fraction"] >= .95)


def summarize_episode(data, active_samples, index, hold_result, required_steps, dt):
    """Select only this episode's original samples, including its terminal one."""
    selected = np.asarray(active_samples, dtype=bool)[:, index]
    episode_data = {
        key: (value[selected] if key == "time_s" else value[selected, index])
        for key, value in data.items()
    }
    count = len(episode_data["time_s"])
    if not count:
        raise ValueError("Each requested episode must have at least one captured step")
    if count != hold_result["steps"][index]:
        raise ValueError("Diagnostic frames do not match frozen hold accumulator")
    steady = episode_data["time_s"] >= 2.0
    steady_summary = summarize_window(episode_data, steady)
    required_steady = int((np.arange(1, required_steps + 1) * dt >= 2.0).sum())
    hold_success = bool(hold_result["success"][index])
    result = {
        "episode_index": index,
        "steps": count,
        "recorded_seconds": float(episode_data["time_s"][-1]),
        "complete_duration": count == required_steps,
        "hold_success": hold_success,
        "strict_stance_success": strict_stance_success(
            hold_success, steady_summary, int(steady.sum()), required_steady),
        "startup_before_2s": summarize_window(episode_data, ~steady),
        "steady_from_2s": steady_summary,
        "all": summarize_window(episode_data, np.ones(count, dtype=bool)),
    }
    for key in ("fallen", "numerical_failure", "early_truncation", "disallowed_support",
                "contact_failure", "max_contact_gap_steps", "max_payload_tilt_deg",
                "max_abs_height_error_m"):
        result[key] = hold_result[key][index]
    return result


def condition_summary(episodes):
    """Count every requested slot; describe means separately from success."""
    def window_average(rows):
        windows = [row["steady_from_2s"] for row in rows if row["steady_from_2s"] is not None]
        if not windows:
            return None
        keys = ("vertical_force_mean_n_left_right", "sole_tilt_mean_deg_left_right",
                "sole_tilt_p95_deg_left_right", "left_load_fraction_mean_when_loaded",
                "both_feet_contact_and_tilt_under_5deg_fraction",
                "both_feet_contact_fraction", "robot_tilt_mean_deg", "joint_velocity_rms_rad_s",
                "joint_torque_rms_nm")
        result = {"episodes_with_steady_samples": len(windows)}
        for key in keys:
            values = [window[key] for window in windows if window[key] is not None]
            result[key] = np.asarray(values).mean(axis=0).tolist() if values else None
        return result

    count = len(episodes)
    return {
        "episodes": count,
        "hold_successes": sum(row["hold_success"] for row in episodes),
        "hold_success_rate": sum(row["hold_success"] for row in episodes) / count,
        "strict_stance_successes": sum(row["strict_stance_success"] for row in episodes),
        "strict_stance_success_rate": sum(row["strict_stance_success"] for row in episodes) / count,
        "completed_duration": sum(row["complete_duration"] for row in episodes),
        "steady_means_all_available_episodes": window_average(episodes),
        "steady_means_hold_successful_episodes": window_average(
            [row for row in episodes if row["hold_success"]]),
        "aggregation_note": "Unweighted average of per-episode steady statistics. Failed/short episodes remain in success denominators; their partial-window means are not full-duration success evidence.",
    }


def evaluate_condition(args, mass_value):
    from lifting_lab import cli
    from lifting_lab.metrics import EpisodeAccumulator
    import torch

    local = copy(args)
    local.mass = mass_value
    demo_args = make_demo_args(local, cli)
    demo_args.num_envs = args.num_envs
    env = None
    try:
        env, mass = cli._env(demo_args, seconds=args.seconds + 1, auto_reset=False,
                             num_envs=args.num_envs)
        policy = cli._policy(env, demo_args)
        obs, _ = env.reset(seed=args.seed)
        dt = float(env.step_dt)
        required = math.ceil(args.seconds / dt)
        if not math.isclose(required * dt, args.seconds, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Evaluation duration must be a whole number of control steps")
        acc = EpisodeAccumulator(args.num_envs, required, support_mode="forearms", contact_grace_steps=6)
        robot = env.scene["robot"]
        sensor = env.scene["feet_ground_contact"]
        if sensor.cfg.reduce != "netforce" or sensor.cfg.num_slots != 1:
            raise ValueError("Expected one world-frame netforce slot per foot")
        names = [name.split("/")[-1] for name in sensor.primary_names]
        if len(names) != 2 or set(names) != {"left_foot_link", "right_foot_link"}:
            raise ValueError(f"Ambiguous foot sensor mapping: {sensor.primary_names}")
        order = [names.index(name) for name in ("left_foot_link", "right_foot_link")]
        sole_ids = robot.find_sites(("left_foot_sole", "right_foot_sole"), preserve_order=True)[0]
        arrays = {key: [] for key in (
            "time_s", "ground_reaction_world_n", "foot_contact_count", "sole_tilt_deg",
            "joint_torque_nm", "joint_vel_rad_s", "robot_tilt_deg")}
        valid_samples = []

        def snapshot(tensor):
            return tensor.detach().cpu().numpy().copy()

        with torch.inference_mode():
            for step in range(required):
                active = acc.active.copy()
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                metrics = cli._metrics(env)
                force = snapshot(sensor.data.force).reshape(args.num_envs, 2, 3)[:, order]
                contact = snapshot(sensor.data.found).reshape(args.num_envs, 2)[:, order]
                quat = snapshot(robot.data.site_quat_w[:, sole_ids])
                # Failure rather than silently computing favorable means from invalid states.
                if not all(np.isfinite(value[active]).all() for value in (force, contact, quat)):
                    raise ValueError("Nonfinite active foot diagnostics; evaluation incomplete")
                if not bool(torch.isfinite(reward[torch.as_tensor(active, device=env.device)]).all()):
                    raise ValueError("Nonfinite active reward; evaluation incomplete")
                sample = {
                    "time_s": (step + 1) * dt,
                    "ground_reaction_world_n": -force,
                    "foot_contact_count": contact,
                    "sole_tilt_deg": sole_tilt_degrees(quat),
                    "joint_torque_nm": snapshot(robot.data.qfrc_actuator),
                    "joint_vel_rad_s": snapshot(robot.data.joint_vel),
                    "robot_tilt_deg": np.asarray(metrics["robot_tilt_deg"]).copy(),
                }
                for key in arrays:
                    arrays[key].append(sample[key])
                valid_samples.append(active)
                acc.update(metrics, snapshot(terminated), snapshot(truncated))
                if not acc.active.any():
                    break
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if reset_ids.numel():
                    obs, _ = env.reset(env_ids=reset_ids)
        data = {key: np.asarray(value) for key, value in arrays.items()}
        valid_samples = np.asarray(valid_samples)
        hold = acc.finish()
        episodes = [summarize_episode(data, valid_samples, i, hold, required, dt)
                    for i in range(args.num_envs)]
        record = {
            "mass": mass, "batch_seed": args.seed, "num_envs": args.num_envs,
            "step_dt_seconds": dt, "required_steps": required,
            "foot_order": ["left", "right"], "sensor_primary_names": sensor.primary_names,
            "joint_names": robot.joint_names,
            "reward_profile": env.cfg.reward_profile,
            "episodes": episodes, "summary": condition_summary(episodes),
            "original_hold_accumulator": hold,
        }
        data["active_original_episode"] = valid_samples
        return record, data
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-envs", type=_positive_int, default=16)
    parser.add_argument("--mass", type=_nonnegative, action="append",
                        help="One kg value per --mass flag; repeat for a sweep")
    parser.add_argument("--seconds", type=_positive, default=20.)
    parser.add_argument("--seed", type=int, default=5011)
    parser.add_argument("--mass-blind", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from lifting_lab import cli
    runtime = cli.runtime_info(args.device)
    masses = args.mass if args.mass is not None else [.4949224923474894]
    metadata_args = copy(args)
    metadata_args.mass = masses[0]
    cli._validate_checkpoint_support(make_demo_args(metadata_args, cli))
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, value in enumerate(masses):
        record, arrays = evaluate_condition(args, value)
        filename = f"condition_{index + 1:02d}_trace.npz"
        with (args.output / filename).open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        record["trace_file"] = filename
        with (args.output / f"condition_{index + 1:02d}.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        results.append(record)
        print(json.dumps({"mass_kg": value, **record["summary"]}), flush=True)
    end_runtime = cli.runtime_info(args.device)
    summary = {
        "schema_version": 1, "protocol_id": "stance-flat5deg-steady95-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": cli.file_sha256(args.checkpoint),
        "runtime": runtime, "runtime_source_sha256_at_end": end_runtime["source_sha256"],
        "source_files_changed_during_evaluation": runtime["source_sha256"] != end_runtime["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostic_helper_sha256": hashlib.sha256(Path(__file__).with_name("diagnose_stance.py").read_bytes()).hexdigest(),
        "seed": args.seed, "batch_num_envs": args.num_envs, "episode_seconds": args.seconds,
        "masses_kg": masses, "conditions": results,
        "hold_success_definition": "Unchanged forearms-contact-v1 EpisodeAccumulator: full duration, box tilt <10deg, abs height error <0.05m, no fall/nonfinite/forbidden contact, support grace 6 control steps.",
        "strict_stance_success_definition": "Additional custom criterion: original hold success AND complete steady window t>=2s AND at least 95% of steady samples have both feet contacting with each sole tilt strictly <5deg. Not a real-robot safety certificate.",
        "sampling": "Post-control-step final physics substep, 50Hz; foot force on robot is negative of foot-primary/terrain-secondary netforce sensor; sole normal independently computed from normalized wxyz quaternion.",
        "limitations": ["One checkpoint/training seed, fixed friction, no pushes or payload-relative jitter.",
            "Same seed, batch size, masses, duration, runtime and source are required for matched comparison.",
            "A contact flag plus low sole tilt does not measure contact area or COP.",
            "Force samples can miss physics-substep peaks; diagnostic means from failed episodes cover shorter periods.",
            "Resets of inactive simulator slots never count as extra episodes; active_original_episode masks the saved trace."]}
    with (args.output / "stance_evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Stance evaluation saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
