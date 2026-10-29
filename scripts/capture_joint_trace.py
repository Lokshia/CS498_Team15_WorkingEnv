"""Capture one non-resetting forearm replay for joint-level diagnosis.

Only load your own or a trusted teammate's checkpoint. This is not training,
formal evaluation, or a stability/safety certification. Example:
python scripts/capture_joint_trace.py --checkpoint RUN/policy.pt --device cpu --output NEW_DIR

NPZ arrays are numeric/Unicode only and load with allow_pickle=False. Joint
state/targets/torques use joint_names order; action uses action_joint_names.
Samples are post-step, including a terminal step. No failed episode is reset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


JOINT_COUNT = 22
ARRAY_FIELDS = ("joint_pos", "joint_vel", "joint_pos_target", "qfrc_actuator", "action")


def _positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return value


def _nonnegative(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative finite number")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Own/trusted team checkpoint with adjacent experiment.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mass", type=_nonnegative, default=0.4949224923474894)
    parser.add_argument("--seed", type=int, default=1011)
    parser.add_argument("--seconds", type=_positive, default=20.0)
    parser.add_argument("--mass-blind", action="store_true", help="Required when replaying a mass-blind checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="New directory; never overwrite or append to a prior capture")
    return parser


def make_demo_args(args, cli_module):
    """Use the project's authoritative parser; do not recreate its env defaults."""
    command = ["demo", "--support-mode", "forearms", "--device", args.device,
               "--num-envs", "1", "--mass", str(args.mass), "--seed", str(args.seed),
               "--seconds", str(args.seconds), "--output", str(args.output)]
    if args.checkpoint is not None:
        command += ["--checkpoint", str(args.checkpoint)]
    if args.mass_blind:
        command.append("--mass-blind")
    return cli_module.build_parser().parse_args(command)


def _names(values, label):
    names = np.asarray(values)
    if names.shape != (JOINT_COUNT,) or names.dtype.kind not in ("U", "S"):
        raise ValueError(f"{label} must contain 22 string names, not object/pickle data")
    names = names.astype(str)
    if len(set(names.tolist())) != JOINT_COUNT or any(not name for name in names):
        raise ValueError(f"{label} must contain 22 unique nonempty names")
    return names


def validate_arrays(arrays):
    """Validate without mutating buffers; return normalised NumPy views/copies."""
    required = ("joint_names", "action_joint_names", "times", *ARRAY_FIELDS)
    for key in required:
        if key not in arrays:
            raise ValueError(f"Missing array {key}")
    result = {"joint_names": _names(arrays["joint_names"], "joint_names"),
              "action_joint_names": _names(arrays["action_joint_names"], "action_joint_names")}
    if set(result["joint_names"]) != set(result["action_joint_names"]):
        raise ValueError("action_joint_names must map exactly to the 22 robot joints")
    times = np.asarray(arrays["times"])
    if times.ndim != 1 or not len(times) or times.dtype.kind not in "fiu":
        raise ValueError("times must be a nonempty numeric vector")
    times = times.astype(float)
    if not np.isfinite(times).all() or times[0] <= 0:
        raise ValueError("times must be finite, positive post-step timestamps")
    if len(times) > 1 and (not (np.diff(times) > 0).all()
                          or not np.allclose(np.diff(times), times[0], rtol=1e-6, atol=1e-9)):
        raise ValueError("times must start at dt and increase uniformly by dt")
    result["times"] = times
    for key in ARRAY_FIELDS:
        values = np.asarray(arrays[key])
        if values.shape != (len(times), JOINT_COUNT) or values.dtype.kind not in "fiu":
            raise ValueError(f"{key} must be a numeric (frames, 22) array")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains nonfinite values")
        result[key] = values.astype(float)
    return result


def _rms(values, axis=None):
    with np.errstate(over="raise", invalid="raise"):
        return np.sqrt(np.mean(np.square(values), axis=axis))


def summarize_arrays(arrays):
    """Per-joint magnitudes, not a stability score. Deltas exclude reset->first."""
    data = validate_arrays(arrays)
    count = len(data["times"])
    names = data["joint_names"].tolist()
    action_names = data["action_joint_names"].tolist()
    indices = [action_names.index(name) for name in names]
    velocities = data["joint_vel"]
    torque = data["qfrc_actuator"]
    velocity_rms = _rms(velocities, axis=0)
    velocity_max = np.abs(velocities).max(axis=0)
    target_delta = np.diff(data["joint_pos_target"], axis=0)
    action_delta = np.diff(data["action"][:, indices], axis=0)
    target_delta_rms = _rms(target_delta, axis=0) if count > 1 else None
    action_delta_rms = _rms(action_delta, axis=0) if count > 1 else None
    torque_rms = _rms(torque, axis=0)
    rows = []
    for index, name in enumerate(names):
        rows.append({"joint": name, "joint_index": index, "action_index": indices[index],
                     "joint_velocity_rms_rad_s": float(velocity_rms[index]),
                     "joint_velocity_max_abs_rad_s": float(velocity_max[index]),
                     "joint_pos_target_delta_rms_rad_per_step": float(target_delta_rms[index]) if count > 1 else None,
                     "action_delta_rms_per_step": float(action_delta_rms[index]) if count > 1 else None,
                     "actuator_torque_rms_nm": float(torque_rms[index]),
                     "actuator_torque_max_abs_nm": float(np.abs(torque[:, index]).max())})
    return {"schema_version": 1, "formal_evaluation": False,
            "scope": "Joint motion diagnosis only; a short or failed capture is not evidence of stability.",
            "frames": count, "simulation_seconds": float(data["times"][-1]),
            "step_dt_seconds": float(data["times"][0]), "difference_samples": count - 1,
            "difference_convention": "Consecutive recorded post-step samples only; excludes reset-to-first action/target.",
            "joint_order": "robot.joint_names", "action_order": "action_joint_names (recorded action-manager order)",
            "joint_velocity_rms_over_all_samples_rad_s": float(_rms(velocities)),
            "action_delta_rms_over_all_samples_per_step": float(_rms(action_delta)) if count > 1 else None,
            "joint_pos_target_delta_rms_over_all_samples_rad_per_step": float(_rms(target_delta)) if count > 1 else None,
            "joints": rows,
            "limitations": ["Only control-step samples are stored; substep peaks and aliasing remain unresolved.",
                            "qfrc_actuator is simulated joint-space actuator torque, not measured real-robot torque.",
                            "Velocity/target/action changes alone do not establish a causal source of oscillation."]}


def save_npz(path, arrays):
    """Exclusive creation, numeric/Unicode arrays only; never pickle objects."""
    clean = validate_arrays(arrays)
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **clean)


def _snapshot(tensor, label):
    values = tensor.detach().cpu().numpy()
    if values.shape != (1, JOINT_COUNT) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite with shape (1, 22)")
    # CPU tensors may alias simulator buffers: capture an immutable-in-time copy.
    return values[0].copy()


def _write_new_json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def capture(args):
    # Heavy dependencies are intentionally lazy; importing this module and its
    # summary functions never loads torch, a checkpoint, or a simulator.
    from lifting_lab import cli
    import torch

    demo_args = make_demo_args(args, cli)
    cli._validate_checkpoint_support(demo_args)
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite capture directory {output}")
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env, mass = cli._env(demo_args, auto_reset=False, num_envs=1, seconds=args.seconds + 1.0)
        if env.num_envs != 1 or env.action_manager.total_action_dim != JOINT_COUNT:
            raise ValueError("Capture requires one environment and exactly 22 actions")
        policy = cli._policy(env, demo_args)
        # The checkpoint wrapper resets during construction; seed only after it.
        obs, _ = env.reset(seed=args.seed)
        robot = env.scene["robot"]
        joint_names = _names(np.asarray(robot.joint_names, dtype=str), "joint_names")
        if env.action_manager.active_terms != ["joint_pos"]:
            raise ValueError("Expected the single joint_pos action term")
        action_names = _names(np.asarray(env.action_manager.get_term("joint_pos").target_names, dtype=str),
                              "action_joint_names")
        if set(action_names) != set(joint_names):
            raise ValueError("Action-to-joint names do not match")
        step_dt = float(env.step_dt)
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Invalid simulator control interval")
        planned_steps = round(args.seconds / step_dt)
        if planned_steps < 1:
            raise ValueError("Requested duration is shorter than one rounded control step")
        samples = {key: [] for key in ARRAY_FIELDS}
        times, trace = [], []
        last_action = _snapshot(env.action_manager.action, "initial action")
        ended_terminated = ended_truncated = False
        with torch.inference_mode():
            for step in range(planned_steps):
                action = policy(obs)
                _snapshot(action, "policy action")
                obs, reward, terminated, truncated, _ = env.step(action)
                if not bool(torch.isfinite(reward).all()) or not all(bool(torch.isfinite(value).all()) for value in obs.values()):
                    raise ValueError("Nonfinite reward or observation; no successful capture can be reported")
                for key in ARRAY_FIELDS:
                    value = env.action_manager.action if key == "action" else getattr(robot.data, key)
                    samples[key].append(_snapshot(value, key))
                metrics = {key: float(value[0]) for key, value in cli._metrics(env).items()}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError("Nonfinite diagnostic metric")
                metrics.setdefault("joint_velocity_rms_rad_s", float(_rms(samples["joint_vel"][-1])))
                metrics.setdefault("action_delta_rms", float(_rms(samples["action"][-1] - last_action)))
                trace.append(metrics)
                last_action = samples["action"][-1]
                times.append((step + 1) * step_dt)
                ended_terminated = bool(terminated.any())
                ended_truncated = bool(truncated.any())
                if ended_terminated or ended_truncated:
                    break  # The terminal sample is kept; never reset or extend it.
        arrays = {key: np.stack(values) for key, values in samples.items()}
        arrays.update(joint_names=joint_names, action_joint_names=action_names, times=np.asarray(times))
        summary = summarize_arrays(arrays)
        stop_reason = ("terminated_and_truncated" if ended_terminated and ended_truncated else
                       "terminated" if ended_terminated else "truncated" if ended_truncated else "requested_step_limit")
        stop = {"requested_seconds": args.seconds, "planned_steps": planned_steps,
                "terminated": ended_terminated, "truncated": ended_truncated,
                "stopped_early": len(trace) < planned_steps, "stop_reason": stop_reason,
                "auto_reset": False, "formal_evaluation": False}
        metadata = {"runtime": cli.runtime_info(env.device), "mass": mass,
                    "randomization": cli._randomization(demo_args),
                    "reward_profile": env.cfg.reward_profile, "reward_profile_schema": 1,
                    "support_mode": env.cfg.support_mode, "seed": args.seed,
                    "mass_observed": not args.mass_blind,
                    "policy": "checkpoint" if args.checkpoint else "zero-action joint-position PD; no trained balancing policy",
                    "checkpoint_sha256": cli.file_sha256(args.checkpoint) if args.checkpoint else None,
                    "frames": len(trace), "simulation_seconds": len(trace) * step_dt,
                    "video": None, "trace": trace, "capture": stop,
                    "joint_trace_file": "joint_trace.npz",
                    "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        summary.update(capture=stop, checkpoint_sha256=metadata["checkpoint_sha256"],
                       runtime_source_sha256=metadata["runtime"]["source_sha256"])
        # Validate JSON serialisability before creating any output payload.
        json.dumps(metadata, allow_nan=False)
        json.dumps(summary, allow_nan=False)
        save_npz(output / "joint_trace.npz", arrays)
        _write_new_json(output / "demo.json", metadata)
        _write_new_json(output / "joint_summary.json", summary)
        print(json.dumps({"output": str(output.resolve()), "frames": len(trace),
                          "simulation_seconds": len(trace) * step_dt, "stop_reason": stop_reason,
                          "formal_evaluation": False}))
        return summary
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    capture(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
