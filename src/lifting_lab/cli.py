"""Small, explicit commands shared by all four teammates."""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

from .runtime import doctor, file_sha256, resolve_device, runtime_info, write_json


def _positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return value


def _nonnegative(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Must be finite and nonnegative")
    return value


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be >= 1")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Booster K1 load-support simulation lab (not a real-robot controller).")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Check pinned runtime and real CUDA availability")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name in ("smoke", "demo", "train", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu" if name == "demo" else "cuda")
        p.add_argument("--num-envs", type=_positive_int, default=1 if name == "demo" else 32)
        p.add_argument("--seed", type=int, default=11)
        mass = p.add_mutually_exclusive_group()
        mass.add_argument("--mass", type=_nonnegative, help="Requested payload mass in kg; 0 uses a documented 1g numerical floor")
        mass.add_argument("--mass-fraction", type=_nonnegative, help="Requested fraction of measured unloaded robot mass, default .025")
        p.add_argument("--mass-blind", action="store_true", help="Remove explicit mass from observations; physics unchanged")
        p.add_argument("--output", type=Path, help="New output directory (never overwrites an existing run)")
        p.add_argument("--threads", type=_positive_int, default=4)
        p.add_argument("--support-mode", choices=("forearms", "pregrasp"), default="forearms",
                       help="forearms: free contact task; pregrasp: legacy CONNECT task")
        if name == "smoke":
            p.add_argument("--steps", type=_positive_int, default=100)
        elif name == "demo":
            p.add_argument("--seconds", type=_positive, default=8)
            p.add_argument("--video", type=Path)
            p.add_argument("--viewer", action="store_true", help="Local Viser browser, no automatic browser launch")
            p.add_argument("--paused", action="store_true", help="Start the viewer at the initial holding pose without advancing physics")
            p.add_argument("--checkpoint", type=Path)
        elif name == "train":
            p.set_defaults(num_envs=512)
            p.add_argument("--iterations", type=_positive_int, default=100, help="PPO iterations per phase")
            p.add_argument("--rollout-steps", type=_positive_int,
                           help="Transitions per environment per PPO iteration; fresh training defaults to 24, resume preserves its prior value")
            p.add_argument("--resume", type=Path, help="Own trusted checkpoint; phase changes can use a different mass")
            p.add_argument("--curriculum", action="store_true", help="Three fixed-mass phases: 0%, 2.5%, 5%; optimizer state is continued")
        else:
            p.add_argument("--protocol", type=Path, help="Versioned protocol; default chosen by support mode")
            p.add_argument("--member", default="member1", help="Label only; does not change seeds or physics")
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--quick", action="store_true", help="2 episodes per condition, 2 seconds, one seed; NOT proposal evaluation")
    return parser


def _output(args, label):
    path = args.output or Path("outputs") / f"{datetime.now():%Y%m%d-%H%M%S-%f}_{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _mass(args):
    from .model import robot_mass_kg
    robot_mass = robot_mass_kg()
    requested = args.mass if args.mass is not None else (args.mass_fraction if args.mass_fraction is not None else 0.025) * robot_mass
    return {"robot_mass_kg": robot_mass, "requested_payload_mass_kg": requested,
            "actual_payload_mass_kg": max(requested, 0.001),
            "requested_mass_fraction": requested / robot_mass,
            "zero_load_uses_1g_proxy": requested < 0.001}


def task_contract(args):
    mode = getattr(args, "support_mode", "pregrasp")
    return {"support_mode": mode,
            "task_schema": "forearms-contact-v1" if mode == "forearms" else "pregrasp-hold-v1",
            "observation_schema": "k1-load-state-97-v1",
            "observation_variant": "blind" if getattr(args, "mass_blind", False) else "conditioned",
            "mass_observed": not getattr(args, "mass_blind", False)}


def _validate_checkpoint_support(args):
    """Read only metadata before constructing a simulator or unpickling weights."""
    checkpoint = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    if checkpoint is None:
        return None
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    path = checkpoint.parent / "experiment.json"
    if not path.is_file():
        raise ValueError("Checkpoint requires its adjacent experiment.json compatibility metadata")
    old = json.loads(path.read_text(encoding="utf-8"))
    expected = task_contract(args)
    # Only the documented, unchanged legacy CONNECT task accepts old metadata.
    if "task_schema" not in old and old.get("support_mode", "pregrasp") == "pregrasp":
        old = {**old, "support_mode": "pregrasp", "task_schema": "pregrasp-hold-v1",
               "observation_schema": "k1-load-state-97-v1",
               "observation_variant": "conditioned" if old.get("mass_observed") else "blind"}
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError(f"Checkpoint {key} differs: {old.get(key)!r} != {value!r}")
    if "runner" not in old:
        raise ValueError("Checkpoint metadata is missing runner configuration")
    if getattr(args, "resume", None) and old.get("seed") != args.seed:
        raise ValueError("Resume must keep the original training seed; it is not an independent seed trial")
    return old


def _env(args, *, seconds=20.0, render=False, auto_reset=True, num_envs=None):
    _validate_checkpoint_support(args)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from .env_cfg import make_env_cfg
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    mass = _mass(args)
    cfg = make_env_cfg(payload_mass_kg=mass["requested_payload_mass_kg"],
                       num_envs=num_envs or args.num_envs, episode_length_s=seconds,
                       mass_observed=not args.mass_blind,
                       support_mode=getattr(args, "support_mode", "pregrasp"))
    cfg.seed = args.seed
    cfg.auto_reset = auto_reset
    if (getattr(args, "viewer", False)
            and getattr(args, "support_mode", "pregrasp") == "forearms"):
        # This is an interactive physics preview, not an evaluation episode.
        # Keep simulating a dropped box until the user resets. Otherwise mjlab
        # correctly rejects the next step after terminal=True, auto_reset=False.
        cfg.terminations = {key: value for key, value in cfg.terminations.items()
                            if key in ("time_out", "nonfinite")}
    if render:
        cfg.viewer.width, cfg.viewer.height = 960, 720
    env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array" if render else None)
    return env, mass


def _metrics(env):
    from .mdp import task_metrics
    return {key: value.detach().cpu().numpy() for key, value in task_metrics(env).items()}


def _runner_config(seed):
    from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
    cfg = RslRlOnPolicyRunnerCfg(seed=seed, logger="tensorboard", upload_model=False,
                               experiment_name="k1_load_balance", clip_actions=1.0)
    cfg.actor = RslRlModelCfg(hidden_dims=(256, 128, 128), obs_normalization=True,
                             distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3})
    cfg.critic = RslRlModelCfg(hidden_dims=(256, 128, 128), obs_normalization=True)
    return asdict(cfg)


def _policy(env, args):
    _validate_checkpoint_support(args)
    import torch
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None:
        return lambda obs: torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    checkpoint = checkpoint.resolve(strict=True)
    experiment = _validate_checkpoint_support(args)
    if experiment.get("support_mode", "pregrasp") != getattr(env.cfg, "support_mode", "pregrasp"):
        raise ValueError("Checkpoint support mode differs from this environment; observation shape alone is not compatibility.")
    if experiment["mass_observed"] != (not args.mass_blind):
        raise ValueError("Checkpoint mass-observation variant differs; use --mass-blind for a blind checkpoint.")
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from tensordict import TensorDict
    wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = MjlabOnPolicyRunner(wrapper, experiment["runner"], device=env.device)
    runner.load(str(checkpoint), map_location=env.device)
    actor = runner.get_inference_policy(device=env.device)
    return lambda obs: actor(TensorDict(obs, batch_size=[env.num_envs])).clamp(-1.0, 1.0)


def smoke(args):
    import torch
    output = _output(args, "smoke")
    env, mass = _env(args)
    try:
        obs, _ = env.reset(seed=args.seed)
        assert env.action_manager.total_action_dim == 22
        assert all(torch.isfinite(value).all() for value in obs.values())
        start = time.perf_counter()
        resets = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                action = torch.zeros((env.num_envs, 22), device=env.device)
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                resets += int((terminated | truncated).sum())
        if env.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = {"passed": True, "scope": "finite physics/API smoke, NOT trained-policy success",
                  "runtime": runtime_info(env.device), "mass": mass,
                  "support_mode": getattr(env.cfg, "support_mode", "pregrasp"),
                  "policy": "zero-action joint-position PD; no trained balancing policy",
                  "num_envs": env.num_envs, "steps_per_env": args.steps, "seed": args.seed,
                  "step_dt": env.step_dt, "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                  "action_dim": 22, "resets": resets, "wall_seconds": elapsed,
                  "transitions_per_second_including_first_step_compile": args.steps * env.num_envs / elapsed,
                  "final_metrics": {k: v.tolist() for k, v in _metrics(env).items()}}
        write_json(output / "smoke.json", result)
        print(f"PASS: physics smoke ({env.device}); {output}")
    finally:
        env.close()


def demo(args):
    _validate_checkpoint_support(args)
    import torch
    import mediapy as media
    if args.viewer and args.video:
        raise ValueError("Choose --viewer OR --video. Viewer has its own Recording panel.")
    if args.video and args.video.exists():
        raise FileExistsError(f"Refusing to overwrite {args.video}")
    output = _output(args, "demo")
    if args.viewer:
        from .viewer import run_interactive
        return run_interactive(args, output, _env, _policy)
    env, mass = _env(args, seconds=args.seconds + 1.0, render=args.video is not None,
                     auto_reset=args.viewer, num_envs=1)
    try:
        policy = _policy(env, args)
        # Wrappers may reset during construction. Apply the agreed seed last.
        obs, _ = env.reset(seed=args.seed)
        metadata = {"runtime": runtime_info(env.device), "mass": mass,
                    "support_mode": getattr(env.cfg, "support_mode", "pregrasp"),
                    "seed": args.seed, "mass_observed": not args.mass_blind,
                    "policy": "checkpoint" if args.checkpoint else "zero-action joint-position PD; no trained balancing policy",
                    "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None}
        writer = None
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = media.VideoWriter(args.video, shape=(720, 960), fps=1.0 / env.step_dt)
            writer.__enter__()
        traces = []
        try:
            with torch.inference_mode():
                for _ in range(round(args.seconds / env.step_dt)):
                    obs, reward, terminated, truncated, _ = env.step(policy(obs))
                    metrics = {k: float(v[0]) for k, v in _metrics(env).items()}
                    traces.append(metrics)
                    if writer:
                        writer.add_image(env.render())
                    if bool((terminated | truncated).any()):
                        break
        finally:
            if writer:
                writer.close()
        write_json(output / "demo.json", {**metadata,
                   "frames": len(traces), "simulation_seconds": len(traces) * env.step_dt,
                   "video": str(args.video) if args.video else None, "trace": traces})
        print(f"Demo complete: {output}; no trained-policy claim implied.")
    finally:
        env.close()


def train(args):
    old = _validate_checkpoint_support(args)
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    output = _output(args, "train")
    fractions = [0.0, 0.025, 0.05] if args.curriculum else [None]
    previous = args.resume.resolve(strict=True) if args.resume else None
    for phase, fraction in enumerate(fractions):
        stage_args = copy(args)
        if fraction is not None:
            stage_args.mass, stage_args.mass_fraction = None, fraction
        elif old and args.mass is None and args.mass_fraction is None:
            # Ordinary resume keeps its previous load unless explicitly changed.
            stage_args.mass = old["mass"]["requested_payload_mass_kg"]
            stage_args.mass_fraction = None
        stage = output / f"phase_{phase}" if args.curriculum else output
        stage.mkdir(parents=True, exist_ok=True)
        # Internal curriculum phases resume the preceding phase, not the original.
        stage_args.resume = previous
        old = _validate_checkpoint_support(stage_args)
        start_iteration, previous_steps = 0, 0
        if previous:
            state = torch.load(previous, map_location="cpu", weights_only=True)
            start_iteration = int(state["iter"]) + 1
            del state
            previous_steps = _checkpoint_steps(old, start_iteration)
        env, mass = _env(stage_args)
        try:
            cfg = deepcopy(old["runner"]) if old else _runner_config(args.seed)
            cfg["max_iterations"] = args.iterations
            cfg["num_steps_per_env"] = _training_rollout_steps(args, old)
            cfg["save_interval"] = min(50, args.iterations)
            phase_steps = args.iterations * cfg["num_steps_per_env"] * env.num_envs
            metadata = {"runtime": runtime_info(env.device), "mass": mass, "seed": args.seed,
                        **task_contract(args), "checkpoint_metadata_schema": 2, "runner": cfg,
                        "episode_seconds": 20, "num_envs": env.num_envs,
                        "phase_start_iteration": start_iteration,
                        "prior_env_steps": previous_steps,
                        "planned_phase_env_steps": phase_steps,
                        "planned_cumulative_env_steps": None if previous_steps is None else previous_steps + phase_steps,
                        "actual_phase_env_steps": None, "actual_cumulative_env_steps": None,
                        "curriculum": bool(args.curriculum), "curriculum_phase_index": phase,
                        "curriculum_mass_fractions": fractions if args.curriculum else None,
                        "phase_history": (old.get("phase_history", []) + [{
                            "checkpoint": str(previous), "mass": old["mass"],
                            "completed_iteration": start_iteration - 1,
                            "cumulative_env_steps": previous_steps}]) if old else [],
                        "resume_from": str(previous) if previous else None,
                        "resume_sha256": file_sha256(previous) if previous else None,
                        "resume_semantics": "model, normalizers and optimizer continued; environment/RNG restart, not bit-exact replay",
                        "task": f"{task_contract(args)['support_mode']} chest-height load support; no pick-up learning"}
            write_json(stage / "experiment.json", metadata)
            runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=1.0), cfg,
                                         log_dir=str(stage), device=env.device)
            # An installed dependency may sit inside an unrelated parent Git
            # repository. Record our source hash above, not that repository's diff.
            runner.logger.git_status_repos = []
            update = runner.alg.update
            def checked_update():
                losses = update()
                _assert_finite_training_state(losses)
                return losses
            runner.alg.update = checked_update
            if previous:
                runner.load(str(previous), map_location=env.device)
                # Upstream stores the last COMPLETED zero-based iteration.
                runner.current_learning_iteration += 1
            runner.learn(args.iterations, init_at_random_ep_len=False)
            _assert_finite_training_state(runner.alg.save())
            actual_iterations = runner.current_learning_iteration - start_iteration + 1
            if actual_iterations != args.iterations:
                raise RuntimeError("Runner completed an unexpected iteration count")
            previous = stage / "policy.pt"
            runner.save(str(previous))
            actual_steps = actual_iterations * cfg["num_steps_per_env"] * env.num_envs
            metadata.update(actual_phase_env_steps=actual_steps,
                            actual_cumulative_env_steps=None if previous_steps is None else previous_steps + actual_steps,
                            actual_iterations_this_phase=actual_iterations,
                            last_completed_iteration=runner.current_learning_iteration)
            write_json(stage / "experiment.json", metadata)
            write_json(stage / "completed.json", {"checkpoint": previous.name,
                       "sha256": file_sha256(previous), "iterations_this_phase": args.iterations,
                       "actual_phase_env_steps": actual_steps,
                       "actual_cumulative_env_steps": metadata["actual_cumulative_env_steps"],
                       "finite_training_state": True,
                       "policy_quality": "not evaluated; a completed train loop is not success evidence"})
        finally:
            env.close()
    print(f"Training loop complete. Evaluate separately. Checkpoint: {previous}")


def _training_rollout_steps(args, previous_metadata):
    override = getattr(args, "rollout_steps", None)
    if override is not None:
        return override
    return previous_metadata["runner"]["num_steps_per_env"] if previous_metadata else 24


def _checkpoint_steps(metadata, next_iteration):
    """Exact transition count for current-schema mid-run and final checkpoints."""
    if "phase_start_iteration" not in metadata:
        return None  # Legacy budgets cannot be reconstructed safely.
    completed = next_iteration - metadata["phase_start_iteration"]
    if completed < 0 or completed * metadata["runner"]["num_steps_per_env"] * metadata["num_envs"] > metadata["planned_phase_env_steps"]:
        raise ValueError("Checkpoint iteration is outside the adjacent metadata's training phase")
    prior = metadata.get("prior_env_steps")
    return None if prior is None else prior + completed * metadata["runner"]["num_steps_per_env"] * metadata["num_envs"]


def _assert_finite_training_state(value):
    import torch
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Nonfinite model/optimizer state: training cannot be marked completed")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Nonfinite training loss/state: training cannot be marked completed")
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_training_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_training_state(item)


def _evaluation_protocol(args):
    protocol_path = args.protocol or (Path(__file__).resolve().parents[2] / "configs" /
        ("team_protocol.json" if task_contract(args)["support_mode"] == "forearms" else "pregrasp_protocol.json"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for key in ("support_mode", "task_schema", "observation_schema"):
        if protocol.get(key) != task_contract(args)[key]:
            raise ValueError(f"Protocol {key} differs from the requested task")
    if task_contract(args)["support_mode"] == "forearms":
        # forearms-contact-v1 environment termination uses six 0.02 s steps.
        # A differently labelled grace cannot override that physics contract.
        grace = protocol["success"].get("contact_grace_seconds")
        if not isinstance(grace, (float, int)) or not math.isclose(grace, 0.12, rel_tol=0, abs_tol=1e-12):
            raise ValueError("forearms-contact-v1 requires contact_grace_seconds=0.12 (6 control steps)")
    return protocol_path, protocol


def evaluate(args):
    experiment = _validate_checkpoint_support(args)
    protocol_path, protocol = _evaluation_protocol(args)
    import numpy as np
    import torch
    from .metrics import EpisodeAccumulator
    output = _output(args, "evaluation")
    fractions = protocol["evaluation_mass_fractions"]
    training_seed = args.seed
    if args.checkpoint:
        training_seed = experiment["seed"]
    if training_seed not in protocol["training_seeds"]:
        raise ValueError("Training seed not in protocol; create a versioned protocol for new seeds.")
    seeds = [protocol["evaluation_seeds"][protocol["training_seeds"].index(training_seed)]]
    seconds = 2.0 if args.quick else protocol["episode_seconds"]
    count = 2 if args.quick else protocol["episodes_per_seed_condition"]
    records = []
    for fraction in fractions:
        for seed in seeds:
            completed = 0
            while completed < count:
                batch = min(args.num_envs, count - completed)
                local = copy(args)
                local.mass, local.mass_fraction, local.seed = None, fraction, seed + completed
                env, mass = _env(local, seconds=seconds, auto_reset=False, num_envs=batch)
                try:
                    policy = _policy(env, local)
                    obs, _ = env.reset(seed=local.seed)
                    required = math.ceil(seconds / env.step_dt)
                    acc = EpisodeAccumulator(batch, required,
                        protocol["success"]["max_payload_tilt_deg_exclusive"],
                        protocol["success"]["max_abs_height_error_m_exclusive"],
                        support_mode=task_contract(args)["support_mode"],
                        contact_grace_steps=round(protocol["success"].get("contact_grace_seconds", 0.12) / env.step_dt))
                    with torch.inference_mode():
                        for _ in range(required):
                            obs, _, terminated, truncated, _ = env.step(policy(obs))
                            acc.update(_metrics(env), terminated.cpu().numpy(), truncated.cpu().numpy())
                            if not acc.active.any():
                                break
                            ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                            if ids.numel():
                                # mjlab requires reset before another step. These
                                # slots remain permanently frozen in acc; their
                                # reset trajectories cannot count as new trials.
                                obs, _ = env.reset(env_ids=ids)
                    record = {"mass": mass, "seed": seed, "batch_seed": local.seed,
                              "episode_offset": completed, "result": acc.finish()}
                    records.append(record)
                    write_json(output / f"condition_{len(records):03d}.json", record)
                finally:
                    env.close()
                completed += batch
    flags = [ok for row in records for ok in row["result"]["success"]]
    summaries = []
    for fraction in fractions:
        condition_flags = [ok for row in records
                           if math.isclose(row["mass"]["requested_mass_fraction"], fraction)
                           for ok in row["result"]["success"]]
        summaries.append({"requested_mass_fraction": fraction,
                          "stress_test": fraction in protocol.get("stress_test_mass_fractions", []),
                          "episodes": len(condition_flags), "successes": sum(condition_flags),
                          "success_rate": float(np.mean(condition_flags))})
    result = {"runtime": runtime_info(resolve_device(args.device)), "protocol": protocol,
              **task_contract(args),
              "protocol_sha256": file_sha256(protocol_path), "member": args.member,
              "quick_smoke_only": args.quick, "episode_seconds": seconds,
              "training_seed": training_seed, "evaluation_seeds_used": seeds,
              "batch_num_envs": args.num_envs,
              "policy": str(args.checkpoint) if args.checkpoint else "zero-action PD baseline (not trained)",
              "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
              "mass_observed": not args.mass_blind, "episodes": len(flags),
              "success_rate": float(np.mean(flags)), "conditions": records,
              "condition_summaries": summaries,
              "limitations": "No-push simulation baseline. Not physical-robot safety validation. Compare same batch size, source fingerprint, protocol and runtime; cross-GPU bitwise equality is not guaranteed."}
    write_json(output / "evaluation.json", result)
    print(f"Evaluation saved: {output}; success={sum(flags)}/{len(flags)}; quick={args.quick}")


def main():
    args = build_parser().parse_args()
    try:
        _validate_checkpoint_support(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.command == "demo" and args.paused and not args.viewer:
        raise SystemExit("--paused requires --viewer")
    if args.command == "train" and args.curriculum and (args.mass is not None or args.mass_fraction is not None):
        raise SystemExit("--curriculum selects masses; do not also specify --mass/--mass-fraction")
    if args.command == "evaluate" and (args.mass is not None or args.mass_fraction is not None):
        raise SystemExit("Evaluation masses come from --protocol; do not override them with --mass")
    os.environ.setdefault("WANDB_MODE", "offline")
    if args.command == "doctor":
        doctor(args.device)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        globals()[args.command](args)


if __name__ == "__main__":
    main()
