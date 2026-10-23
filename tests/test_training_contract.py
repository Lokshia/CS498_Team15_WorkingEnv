"""Fast metadata, budget and protocol regressions; never starts a simulator."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lifting_lab import cli


class TrainingContractTests(unittest.TestCase):
    def test_rollout_steps_are_explicit(self):
        args = cli.build_parser().parse_args(["train", "--iterations", "3", "--num-envs", "32", "--rollout-steps", "8"])
        self.assertEqual(args.rollout_steps, 8)
        self.assertEqual(args.support_mode, "forearms")

    def test_fresh_training_defaults_to_24_rollout_steps(self):
        args = cli.build_parser().parse_args(["train"])
        self.assertIsNone(args.rollout_steps)
        self.assertEqual(cli._training_rollout_steps(args, None), 24)

    def test_resume_preserves_previous_eight_step_rollout_without_override(self):
        args = cli.build_parser().parse_args(["train", "--resume", "policy.pt"])
        self.assertEqual(cli._training_rollout_steps(args, {"runner": {"num_steps_per_env": 8}}), 8)

    def test_explicit_resume_rollout_override_wins(self):
        args = cli.build_parser().parse_args(["train", "--resume", "policy.pt", "--rollout-steps", "12"])
        self.assertEqual(cli._training_rollout_steps(args, {"runner": {"num_steps_per_env": 8}}), 12)

    def checkpoint(self, directory, mode="forearms", **changes):
        path = Path(directory) / "policy.pt"
        path.write_bytes(b"metadata-only fixture")
        args = SimpleNamespace(support_mode=mode, mass_blind=False)
        metadata = {**cli.task_contract(args), "runner": {}, "seed": 11}
        metadata.update(changes)
        (path.parent / "experiment.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def test_matching_forearm_checkpoint_is_permitted_without_loading_weights(self):
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory)
            args = SimpleNamespace(checkpoint=path, support_mode="forearms", mass_blind=False)
            self.assertEqual(cli._validate_checkpoint_support(args)["task_schema"], "forearms-contact-v1")

    def test_each_mismatched_schema_and_variant_is_rejected_before_environment(self):
        for key, value in {"support_mode": "pregrasp", "task_schema": "old-task", "observation_schema": "other-97",
                           "observation_variant": "blind", "mass_observed": False}.items():
            with self.subTest(key=key), TemporaryDirectory() as directory:
                path = self.checkpoint(directory, **{key: value})
                args = SimpleNamespace(checkpoint=path, support_mode="forearms", mass_blind=False)
                with patch.object(cli, "_mass") as mass, self.assertRaisesRegex(ValueError, key):
                    cli._env(args)
                mass.assert_not_called()

    def test_forearm_missing_task_schema_fails(self):
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory)
            meta_path = path.parent / "experiment.json"
            meta = json.loads(meta_path.read_text())
            del meta["task_schema"]
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "task_schema"):
                cli._validate_checkpoint_support(SimpleNamespace(checkpoint=path, support_mode="forearms", mass_blind=False))

    def test_legacy_pregrasp_requires_explicit_mode_and_cannot_cross_to_forearms(self):
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory, mode="pregrasp")
            (path.parent / "experiment.json").write_text(json.dumps({"runner": {}, "seed": 11, "mass_observed": True}))
            cli._validate_checkpoint_support(SimpleNamespace(checkpoint=path, support_mode="pregrasp", mass_blind=False))
            with self.assertRaisesRegex(ValueError, "support_mode"):
                cli._validate_checkpoint_support(SimpleNamespace(checkpoint=path, support_mode="forearms", mass_blind=False))

    def test_resume_must_preserve_training_seed(self):
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory)
            with self.assertRaisesRegex(ValueError, "training seed"):
                cli._validate_checkpoint_support(SimpleNamespace(resume=path, support_mode="forearms", mass_blind=False, seed=22))

    def test_midphase_budget_uses_saved_iteration_not_planned_total(self):
        metadata = {"phase_start_iteration": 3, "prior_env_steps": 2304,
                    "runner": {"num_steps_per_env": 24}, "num_envs": 32,
                    "planned_phase_env_steps": 1536}
        self.assertEqual(cli._checkpoint_steps(metadata, 4), 3072)
        self.assertEqual(cli._checkpoint_steps(metadata, 5), 3840)
        for iteration in (2, 6):
            with self.assertRaises(ValueError):
                cli._checkpoint_steps(metadata, iteration)
        self.assertIsNone(cli._checkpoint_steps({}, 100))

    def test_nonfinite_optimizer_or_policy_prevents_completed_status(self):
        cli._assert_finite_training_state({"actor": torch.ones(2), "optimizer": [{"step": torch.tensor(3)}]})
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            cli._assert_finite_training_state({"optimizer": [{"moment": torch.tensor(float("nan"))}]})
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            cli._assert_finite_training_state({"policy_loss": float("inf")})

    def test_protocols_are_distinct_and_formal(self):
        root = Path(cli.__file__).resolve().parents[2] / "configs"
        for mode, name in (("forearms", "team_protocol.json"), ("pregrasp", "pregrasp_protocol.json")):
            protocol = json.loads((root / name).read_text())
            self.assertEqual(protocol["support_mode"], mode)
            self.assertEqual(protocol["episode_seconds"], 20)
            self.assertEqual(protocol["episodes_per_seed_condition"], 50)
            self.assertEqual(protocol["training_seeds"], [11, 22, 33])

    def test_protocol_grace_must_match_environment_v1(self):
        args = cli.build_parser().parse_args(["evaluate"])
        _, protocol = cli._evaluation_protocol(args)
        with TemporaryDirectory() as directory:
            args.protocol = Path(directory) / "protocol.json"
            for grace in (0.0, 0.24, None, "0.12"):
                with self.subTest(grace=grace):
                    protocol["success"]["contact_grace_seconds"] = grace
                    args.protocol.write_text(json.dumps(protocol))
                    with self.assertRaisesRegex(ValueError, "contact_grace_seconds"):
                        cli._evaluation_protocol(args)


if __name__ == "__main__":
    unittest.main()
