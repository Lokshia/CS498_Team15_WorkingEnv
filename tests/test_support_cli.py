"""Support-mode routing and checkpoint isolation without training or a server."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab import cli
from lifting_lab.viewer import prepare_world, world_snapshot


class SupportParserTests(unittest.TestCase):
    def test_all_commands_default_to_forearms(self):
        for command in ("demo", "smoke", "train", "evaluate"):
            with self.subTest(command=command):
                args = cli.build_parser().parse_args([command])
                self.assertEqual(args.support_mode, "forearms")

    def test_legacy_pregrasp_remains_explicitly_selectable(self):
        for command in ("demo", "smoke", "train", "evaluate"):
            with self.subTest(command=command):
                args = cli.build_parser().parse_args([command, "--support-mode", "pregrasp"])
                self.assertEqual(args.support_mode, "pregrasp")

    def test_train_and_evaluate_can_select_forearms(self):
        for command in ("train", "evaluate"):
            with self.subTest(command=command):
                args = cli.build_parser().parse_args([command])
                self.assertEqual(args.support_mode, "forearms")
                self.assertEqual(cli.build_parser().parse_args([command, "--support-mode", "forearms"]).support_mode, "forearms")


class CheckpointIsolationTests(unittest.TestCase):
    def test_missing_checkpoint_is_rejected_before_main_dispatch(self):
        with patch.object(sys, "argv", ["lifting_lab", "demo", "--viewer", "--paused",
                                        "--checkpoint", "does-not-need-to-exist.pt"]):
            with patch.object(cli, "demo") as demo:
                with self.assertRaisesRegex(SystemExit, "Checkpoint does not exist"):
                    cli.main()
                demo.assert_not_called()

    def test_direct_demo_rejects_checkpoint_before_output_or_environment(self):
        args = SimpleNamespace(support_mode="forearms", checkpoint=Path("policy.pt"))
        with patch.object(cli, "_output") as output, patch.object(cli, "_env") as env:
            with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
                cli.demo(args)
            output.assert_not_called()
            env.assert_not_called()

    def test_viewer_preparation_rejects_checkpoint_before_environment_factory(self):
        args = SimpleNamespace(support_mode="forearms", checkpoint=Path("policy.pt"))
        factory, policy = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
            prepare_world(args, factory, policy)
        factory.assert_not_called()
        policy.assert_not_called()

    def test_direct_environment_rejects_checkpoint_before_mass_or_device_work(self):
        args = SimpleNamespace(support_mode="forearms", checkpoint=Path("policy.pt"))
        with patch.object(cli, "_mass") as mass, patch.object(cli, "resolve_device") as device:
            with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
                cli._env(args)
            mass.assert_not_called()
            device.assert_not_called()

    def test_missing_mode_on_legacy_namespace_is_pregrasp(self):
        self.assertEqual(cli.task_contract(SimpleNamespace())["support_mode"], "pregrasp")
        cli._validate_checkpoint_support(SimpleNamespace(support_mode="forearms", checkpoint=None))

    def test_saved_support_mode_mismatch_is_rejected_before_runner_construction(self):
        with TemporaryDirectory(prefix="lifting-support-checkpoint-") as directory:
            checkpoint = Path(directory) / "policy.pt"
            checkpoint.write_bytes(b"not-a-real-checkpoint")
            (checkpoint.parent / "experiment.json").write_text(
                json.dumps({"support_mode": "forearms", "mass_observed": True}), encoding="utf-8")
            args = SimpleNamespace(support_mode="pregrasp", checkpoint=checkpoint, mass_blind=False)
            env = SimpleNamespace(cfg=SimpleNamespace(support_mode="pregrasp"))
            with self.assertRaisesRegex(ValueError, "Checkpoint support_mode differs"):
                cli._policy(env, args)


class SupportForwardingTests(unittest.TestCase):
    def test_environment_forwards_explicit_mode_and_legacy_fallback(self):
        for mode in (None, "pregrasp", "forearms"):
            with self.subTest(mode=mode):
                args = SimpleNamespace(device="cpu", threads=1, num_envs=1, mass_blind=False, seed=123)
                if mode is not None:
                    args.support_mode = mode
                cfg = SimpleNamespace()
                mass = {"requested_payload_mass_kg": .5}
                make_cfg = Mock(return_value=cfg)
                with patch.object(cli, "_mass", return_value=mass):
                    with patch.dict(sys.modules, {"lifting_lab.env_cfg": SimpleNamespace(make_env_cfg=make_cfg)}):
                        with patch("mjlab.envs.ManagerBasedRlEnv") as constructor:
                            env, actual_mass = cli._env(args)
                self.assertIs(env, constructor.return_value)
                self.assertIs(actual_mass, mass)
                self.assertEqual(make_cfg.call_args.kwargs["support_mode"], mode or "pregrasp")
                self.assertEqual(cfg.seed, 123)

    def test_world_snapshot_identifies_free_contact_mode_and_zero_equalities(self):
        model = SimpleNamespace(
            body=Mock(return_value=SimpleNamespace(id=1)), neq=0,
            body_mass=np.array([0.0, .5]),
            body_inertia=np.array([[0.0, 0.0, 0.0], [.01, .02, .03]]),
        )
        env = SimpleNamespace(
            cfg=SimpleNamespace(support_mode="forearms"),
            sim=SimpleNamespace(mj_model=model,
                                model=SimpleNamespace(body_mass=torch.tensor([0.0, .5])),
                                data=SimpleNamespace(time=torch.tensor([0.0]))),
            episode_length_buf=torch.tensor([0]),
        )
        snapshot = world_snapshot(env, {"actual_payload_mass_kg": .5}, 0, True)
        self.assertEqual(snapshot["support_mode"], "forearms")
        self.assertEqual(snapshot["equality_constraints"], 0)
        self.assertEqual(snapshot["simulation_time_s"], 0.0)
        self.assertEqual(snapshot["control_steps"], 0)
        json.dumps(snapshot, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
