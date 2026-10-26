"""Explicit, bounded end-to-end acceptance test (NOT a policy-quality benchmark).

Run with this project's Python: scripts/check_training.py --device cuda
Creates a new output directory. Never installs dependencies or starts a viewer.
Only checkpoints produced by this invocation are loaded by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.num_envs <= 64:
        parser.error("This bounded test requires 2–64 environments; use train for larger runs.")
    output = (args.output or ROOT / "outputs" / f"{datetime.now():%Y%m%d-%H%M%S-%f}_training_acceptance").resolve()
    output.mkdir(parents=True, exist_ok=False)
    from lifting_lab.runtime import runtime_info, resolve_device, write_json
    import torch

    started = time.perf_counter()
    report = {"passed": False, "scope": "training/save/resume/curriculum/evaluation/inference plumbing only; NOT learned balance",
              "runtime": runtime_info(resolve_device(args.device)), "commands": [], "checkpoints": [],
              "num_envs": args.num_envs, "rollout_steps": 24}
    common = ["--support-mode", "forearms", "--device", args.device, "--num-envs", str(args.num_envs)]

    def run(label, command):
        argv = [sys.executable, "-u", "-m", "lifting_lab", *command]
        print(f"CHECK {label}", flush=True)
        with (output / f"{label}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        report["commands"].append({"label": label, "argv": argv, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {output / (label + '.log')}")

    def checkpoint(folder, expected_iter):
        path = output / folder / "policy.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tensor_count = 0

        def finite(value):
            nonlocal tensor_count
            if isinstance(value, torch.Tensor):
                tensor_count += 1
                if not torch.isfinite(value).all():
                    raise AssertionError(f"Nonfinite checkpoint tensor: {path}")
            elif isinstance(value, dict):
                for child in value.values():
                    finite(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite(child)

        finite(saved)
        assert saved["iter"] == expected_iter, (path, saved["iter"], expected_iter)
        assert saved["optimizer_state_dict"]["state"], "No optimizer state was saved"
        meta = json.loads((path.parent / "experiment.json").read_text(encoding="utf-8"))
        assert meta["support_mode"] == "forearms"
        assert meta["num_envs"] == args.num_envs
        assert meta["runner"]["num_steps_per_env"] == 24
        assert (path.parent / "completed.json").is_file()
        report["checkpoints"].append({"folder": folder, "iteration": expected_iter,
                                      "finite_tensors": tensor_count, "mass": meta["mass"]})
        return saved

    try:
        run("conditioned", ["train", *common, "--seed", "11", "--iterations", "3",
                            "--output", str(output / "conditioned")])
        first = checkpoint("conditioned", 2)
        run("resumed", ["train", *common, "--seed", "11", "--iterations", "2",
                        "--resume", str(output / "conditioned/policy.pt"), "--output", str(output / "resumed")])
        resumed = checkpoint("resumed", 4)
        weights = [name for name in first["actor_state_dict"] if name.startswith("mlp.") and name.endswith("weight")]
        assert weights, "No actor network weights found"
        report["actor_weights_changed_after_resume"] = any(
            not torch.equal(first["actor_state_dict"][name], resumed["actor_state_dict"][name]) for name in weights)
        assert report["actor_weights_changed_after_resume"]
        old_optimizer = first["optimizer_state_dict"]["state"]
        new_optimizer = resumed["optimizer_state_dict"]["state"]
        # A fresh optimizer would restart these counters; require continuation
        # on every shared parameter, not merely the existence of saved moments.
        assert old_optimizer.keys() == new_optimizer.keys()
        report["optimizer_steps_continued"] = all(
            float(new_optimizer[key]["step"]) > float(old_optimizer[key]["step"])
            for key in old_optimizer)
        assert report["optimizer_steps_continued"]
        run("curriculum", ["train", *common, "--seed", "11", "--iterations", "1", "--curriculum",
                           "--output", str(output / "curriculum")])
        for i in range(3):
            checkpoint(f"curriculum/phase_{i}", i)
        run("blind", ["train", *common, "--seed", "11", "--iterations", "2", "--mass-blind",
                      "--output", str(output / "blind")])
        checkpoint("blind", 1)
        randomized = ["--mass-range-fraction", "0", "0.05", "--com-offset-m", "0.02", "0.03", "0",
                      "--payload-xy-jitter-m", "0.01", "--payload-yaw-jitter-rad", "0.02",
                      "--push-interval-s", "0.5", "1.0", "--push-velocity-m-s", "0.3"]
        run("randomized", ["train", *common, "--seed", "11", "--iterations", "2", *randomized,
                           "--output", str(output / "randomized")])
        saved = checkpoint("randomized", 1)
        meta = json.loads((output / "randomized" / "experiment.json").read_text(encoding="utf-8"))
        assert meta["mass"]["mass_range_fraction"] == [0.0, 0.05]
        assert meta["randomization"]["push_velocity_m_s"] == 0.3
        assert meta["task_schema"] == "forearms-contact-v1"
        run("randomized_resumed", ["train", *common, "--seed", "11", "--iterations", "1",
                                   "--resume", str(output / "randomized/policy.pt"),
                                   "--output", str(output / "randomized_resumed")])
        checkpoint("randomized_resumed", 2)
        meta = json.loads((output / "randomized_resumed" / "experiment.json").read_text(encoding="utf-8"))
        assert meta["randomization"]["push_interval_s"] == [0.5, 1.0], "resume lost the push settings"
        assert meta["mass"]["mass_range_fraction"] == [0.0, 0.05], "resume lost the mass range"
        report["randomized_training"] = {"iterations": 3, "randomization": meta["randomization"],
                                         "finite_tensors": len(saved["actor_state_dict"])}
        for label, model, blind, protocol in (("eval_conditioned", "resumed/policy.pt", False, None),
                                              ("eval_blind", "blind/policy.pt", True, None),
                                              ("eval_pd_baseline", None, False, None),
                                              ("eval_push", "randomized_resumed/policy.pt", False,
                                               ROOT / "configs" / "team_protocol_push.json")):
            command = ["evaluate", *common, "--quick", "--output", str(output / label)]
            if model:
                command += ["--checkpoint", str(output / model)]
            if blind:
                command += ["--mass-blind"]
            if protocol:
                command += ["--protocol", str(protocol)]
            run(label, command)
            evaluation = json.loads((output / label / "evaluation.json").read_text(encoding="utf-8"))
            assert evaluation["support_mode"] == "forearms" and evaluation["quick_smoke_only"]
            assert evaluation["episodes"] == 6 and evaluation["episode_seconds"] == 2.0
            assert (evaluation["disturbance"] is not None) == (protocol is not None)
            report[label] = {"episodes": evaluation["episodes"], "success_rate": evaluation["success_rate"],
                             "protocol_id": evaluation["protocol"]["protocol_id"],
                             "policy_quality_claim": False}
        run("cpu_checkpoint_demo", ["demo", "--support-mode", "forearms", "--device", "cpu", "--seconds", "1",
                                    "--checkpoint", str(output / "resumed/policy.pt"),
                                    "--output", str(output / "cpu_checkpoint_demo")])
        demo = json.loads((output / "cpu_checkpoint_demo/demo.json").read_text(encoding="utf-8"))
        assert demo["frames"] > 0 and demo["policy"] == "checkpoint"
        report["cpu_inference_frames"] = demo["frames"]
        # Exercise the viewer's world/policy path without opening a server or
        # disturbing the user's already-running viewer. Preview must continue
        # after a deliberately dropped box; it must NOT turn that into success.
        from lifting_lab import cli
        from lifting_lab.viewer import prepare_world, world_snapshot
        viewer_checks = []
        for mass in (0.5, 2.0):
            view_args = cli.build_parser().parse_args([
                "demo", "--device", "cpu", "--support-mode", "forearms",
                "--viewer", "--paused", "--seconds", "2", "--mass", str(mass),
                "--checkpoint", str(output / "resumed/policy.pt")])
            env, mass_info, policy, wrapper = prepare_world(view_args, cli._env, cli._policy)
            try:
                snap = world_snapshot(env, mass_info, revision=0, paused=True)
                assert snap["simulation_time_s"] == 0 and snap["equality_constraints"] == 0
                assert abs(snap["compiled_payload_mass_kg"] - mass) < 1e-7
                obs = wrapper.get_observations()
                with torch.inference_mode():
                    first_action, second_action = policy(obs), policy(obs)
                    assert torch.equal(first_action, second_action), "Inference is not deterministic"
                    assert torch.isfinite(first_action).all()
                    payload = env.scene["payload"]
                    state = payload.data.default_root_state.clone()
                    state[:, :3] += env.scene.env_origins
                    state[:, 2] = env.scene.env_origins[:, 2] + 0.08
                    payload.write_root_state_to_sim(state)
                    for _ in range(25):
                        obs, reward, terminated, truncated, _ = env.step(policy(obs))
                        assert not bool((terminated | truncated).any()), "Preview stopped on task failure"
                        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
                viewer_checks.append({"requested_mass_kg": mass, "finite_steps_after_forced_drop": 25,
                                      "deterministic_inference": True, "opened_browser_or_server": False})
            finally:
                env.close()
        report["viewer_world_checks"] = viewer_checks
        report["passed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        write_json(output / "acceptance.json", report)
        print(f"Acceptance report: {output / 'acceptance.json'}", flush=True)
    print("PASS: bounded training plumbing. No balance-performance claim.", flush=True)


if __name__ == "__main__":
    main()
