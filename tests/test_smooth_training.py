"""Reward-only profile routing, explicit training overrides and diagnostic invariants."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifting_lab import cli
from lifting_lab.metrics import EpisodeAccumulator


class RewardProfileTests(unittest.TestCase):
    def args(self, command="train", *extra):
        return cli.build_parser().parse_args([command, *extra])

    def metadata(self, **changes):
        result = {**cli.task_contract(self.args()), "runner": {}, "seed": 11}
        result.update(changes)
        return result

    def test_all_commands_have_unset_profile_and_fresh_baseline(self):
        for command in ("smoke", "demo", "train", "evaluate"):
            args = self.args(command)
            self.assertIsNone(args.reward_profile)
            self.assertEqual(cli.reward_profile(args), "baseline")

    def test_fresh_forearms_can_select_smooth_profile(self):
        for command in ("smoke", "demo", "train", "evaluate"):
            args = self.args(command, "--reward-profile", "smooth-v1")
            self.assertEqual(cli.reward_profile(args), "smooth-v1")

    def test_pregrasp_cannot_select_smooth_profile_before_environment_creation(self):
        args = self.args("smoke", "--support-mode", "pregrasp", "--reward-profile", "smooth-v1")
        with patch.object(cli, "_mass") as mass, self.assertRaisesRegex(ValueError, "only supported"):
            cli._env(args)
        mass.assert_not_called()

    def test_missing_legacy_profile_is_baseline(self):
        self.assertEqual(cli.reward_profile(self.args(), self.metadata()), "baseline")

    def test_checkpoint_and_resume_inherit_saved_smooth_profile(self):
        for command, option in (("demo", "--checkpoint"), ("evaluate", "--checkpoint"), ("train", "--resume")):
            args = self.args(command, option, "policy.pt")
            self.assertEqual(cli.reward_profile(args, self.metadata(reward_profile="smooth-v1", reward_profile_schema=1)), "smooth-v1")

    def test_explicit_training_profile_change_is_allowed(self):
        args = self.args("train", "--resume", "policy.pt", "--reward-profile", "smooth-v1")
        self.assertEqual(cli.reward_profile(args, self.metadata()), "smooth-v1")

    def test_explicit_demo_or_evaluation_mismatch_is_rejected(self):
        for command in ("demo", "evaluate"):
            args = self.args(command, "--checkpoint", "policy.pt", "--reward-profile", "baseline")
            with self.assertRaisesRegex(ValueError, "must match"):
                cli.reward_profile(args, self.metadata(reward_profile="smooth-v1", reward_profile_schema=1))

    def test_unsupported_schema_and_unknown_profile_fail(self):
        for changes in ({"reward_profile": "unknown"},
                        {"reward_profile": "smooth-v1"},
                        {"reward_profile": "smooth-v1", "reward_profile_schema": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cli.reward_profile(self.args(), self.metadata(**changes))

    def test_validation_reads_profile_before_building_or_loading_weights(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            checkpoint.write_bytes(b"metadata-only fixture, never loaded")
            metadata = self.metadata(reward_profile="smooth-v1", reward_profile_schema=1)
            (checkpoint.parent / "experiment.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = self.args("evaluate", "--checkpoint", str(checkpoint))
            self.assertEqual(cli._validate_checkpoint_support(args)["reward_profile"], "smooth-v1")
            args.reward_profile = "baseline"
            with patch.object(cli, "_mass") as mass, self.assertRaisesRegex(ValueError, "must match"):
                cli._env(args)
            mass.assert_not_called()


class TrainingOverrideTests(unittest.TestCase):
    def test_unset_overrides_do_not_change_saved_ppo_config(self):
        cfg = {"algorithm": {"learning_rate": .0007, "entropy_coef": .005, "schedule": "adaptive"}}
        before = deepcopy(cfg)
        cli._configure_training_overrides(cfg, cli.build_parser().parse_args(["train"]))
        self.assertEqual(cfg, before)

    def test_explicit_ppo_overrides_allow_zero_entropy(self):
        args = cli.build_parser().parse_args(["train", "--learning-rate", ".0003", "--entropy-coef", "0"])
        cfg = {"algorithm": {"learning_rate": .001, "entropy_coef": .005}}
        cli._configure_training_overrides(cfg, args)
        self.assertEqual(cfg["algorithm"], {"learning_rate": .0003, "entropy_coef": 0.0})

    def test_lr_override_changes_all_restored_groups_but_keeps_optimizer_state(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": .001}, {"lr": .002}], state={"momentum": [1, 2]})
        runner = SimpleNamespace(alg=SimpleNamespace(learning_rate=.001, optimizer=optimizer))
        cli._apply_learning_rate_override(runner, SimpleNamespace(learning_rate=.0003))
        self.assertEqual(runner.alg.learning_rate, .0003)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [.0003, .0003])
        self.assertEqual(optimizer.state, {"momentum": [1, 2]})

    def test_unset_lr_keeps_restored_optimizer_rate(self):
        runner = SimpleNamespace(alg=SimpleNamespace(learning_rate=.0007,
                                 optimizer=SimpleNamespace(param_groups=[{"lr": .0007}])))
        cli._apply_learning_rate_override(runner, SimpleNamespace(learning_rate=None))
        self.assertEqual(runner.alg.learning_rate, .0007)
        self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], .0007)

    def test_invalid_optimizer_overrides_are_parse_errors(self):
        for option, values in (("--learning-rate", ("0", "-1", "nan", "inf")),
                               ("--entropy-coef", ("-1", "nan", "inf"))):
            for value in values:
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.build_parser().parse_args(["train", option, value])

    def test_optimizer_override_flags_are_training_only(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["demo", "--entropy-coef", "0"])


class SmoothnessDiagnosticTests(unittest.TestCase):
    def metrics(self, delta):
        return {"payload_tilt_deg": np.array([0.0]), "height_error_m": np.array([0.0]),
                "action_delta_rms": np.array([delta]), "airborne": np.array([1.0]),
                "joint_velocity_rms_rad_s": np.array([2.0])}

    def test_diagnostics_do_not_redefine_formal_success(self):
        acc = EpisodeAccumulator(1, 2)
        acc.update(self.metrics(.2), [False])
        acc.update(self.metrics(.4), [False])
        result = acc.finish()
        self.assertEqual(result["success"], [True])
        self.assertNotIn("steady_hold_success", result)
        means = result["smoothness_summary"]["per_episode_means"]
        self.assertAlmostEqual(means["action_delta_rms"][0], .3)
        self.assertEqual(means["airborne"], [1.0])

    def test_frozen_episode_diagnostics_cannot_change_after_reset(self):
        acc = EpisodeAccumulator(1, 3)
        acc.update(self.metrics(.2), [True])
        before = acc.finish()
        acc.update(self.metrics(100), [False])
        self.assertEqual(acc.finish(), before)

    def test_legacy_metrics_produce_empty_optional_summary(self):
        acc = EpisodeAccumulator(1, 1)
        acc.update({"payload_tilt_deg": np.array([0.0]), "height_error_m": np.array([0.0])}, [False])
        self.assertEqual(acc.finish()["smoothness_summary"]["per_episode_means"], {})


if __name__ == "__main__":
    unittest.main()
