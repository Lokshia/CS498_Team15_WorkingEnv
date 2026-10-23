"""CPU-only CLI contracts and subprocess tests for result compatibility.

No command in this file constructs an environment or starts training. Evaluation
files are synthetic fixtures in an automatically cleaned temporary directory.
"""
from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifting_lab import cli


class ParserTests(unittest.TestCase):
    def test_viewer_can_start_paused_without_a_checkpoint(self):
        args = cli.build_parser().parse_args(["demo", "--viewer", "--paused", "--mass", "0.5"])
        self.assertTrue(args.viewer)
        self.assertTrue(args.paused)
        self.assertIsNone(args.checkpoint)
        self.assertFalse(cli.build_parser().parse_args(["demo", "--viewer"]).paused)

    def assert_parse_rejected(self, arguments):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.build_parser().parse_args(arguments)
        self.assertEqual(raised.exception.code, 2)

    def test_negative_and_nonfinite_mass_are_rejected(self):
        for command in ("smoke", "demo", "train", "evaluate"):
            for option in ("--mass", "--mass-fraction"):
                for value in ("-0.01", "nan", "inf", "-inf"):
                    with self.subTest(command=command, option=option, value=value):
                        self.assert_parse_rejected([command, f"{option}={value}"])

    def test_steps_must_be_a_positive_integer(self):
        for value in ("0", "-1", "1.5", "nan", "inf", "invalid"):
            with self.subTest(value=value):
                self.assert_parse_rejected(["smoke", f"--steps={value}"])

    def test_absolute_and_relative_mass_are_mutually_exclusive(self):
        for command in ("smoke", "demo", "train", "evaluate"):
            with self.subTest(command=command):
                self.assert_parse_rejected([command, "--mass", "1", "--mass-fraction", ".05"])

    def test_training_smoke_interface_is_parseable(self):
        args = cli.build_parser().parse_args([
            "train", "--device", "cpu", "--iterations", "1", "--num-envs", "2",
            "--seed", "22", "--mass-fraction", ".025", "--mass-blind",
            "--output", "outputs/test-only", "--threads", "1",
        ])
        self.assertEqual((args.command, args.device, args.iterations, args.num_envs),
                         ("train", "cpu", 1, 2))
        self.assertEqual((args.seed, args.mass_fraction, args.threads), (22, .025, 1))
        self.assertTrue(args.mass_blind)
        self.assertEqual(args.output, Path("outputs/test-only"))


class MassConversionTests(unittest.TestCase):
    def mass_for(self, *arguments):
        args = cli.build_parser().parse_args(["smoke", *arguments])
        # A known robot mass isolates kg/fraction conversion from MJCF loading.
        with patch("lifting_lab.model.robot_mass_kg", return_value=20.0):
            return cli._mass(args)

    def test_default_is_two_and_a_half_percent(self):
        mass = self.mass_for()
        self.assertEqual(mass["robot_mass_kg"], 20.0)
        self.assertEqual(mass["requested_mass_fraction"], .025)
        self.assertEqual(mass["requested_payload_mass_kg"], .5)
        self.assertEqual(mass["actual_payload_mass_kg"], .5)
        self.assertFalse(mass["zero_load_uses_1g_proxy"])

    def test_zero_request_is_preserved_separately_from_one_gram_floor(self):
        for option in ("--mass", "--mass-fraction"):
            with self.subTest(option=option):
                mass = self.mass_for(option, "0")
                self.assertEqual(mass["requested_payload_mass_kg"], 0.0)
                self.assertEqual(mass["requested_mass_fraction"], 0.0)
                self.assertEqual(mass["actual_payload_mass_kg"], .001)
                self.assertTrue(mass["zero_load_uses_1g_proxy"])

    def test_explicit_kilograms_convert_to_fraction(self):
        mass = self.mass_for("--mass", "1.75")
        self.assertEqual(mass["actual_payload_mass_kg"], 1.75)
        self.assertAlmostEqual(mass["requested_mass_fraction"], 1.75 / 20.0)

    def test_explicit_fraction_converts_to_kilograms(self):
        mass = self.mass_for("--mass-fraction", ".075")
        self.assertEqual(mass["requested_mass_fraction"], .075)
        self.assertEqual(mass["actual_payload_mass_kg"], 1.5)

    def test_subgram_request_also_exposes_actual_numerical_floor(self):
        mass = self.mass_for("--mass", ".0005")
        self.assertEqual(mass["requested_payload_mass_kg"], .0005)
        self.assertEqual(mass["actual_payload_mass_kg"], .001)
        self.assertTrue(mass["zero_load_uses_1g_proxy"])


class MainValidationTests(unittest.TestCase):
    def test_paused_requires_interactive_viewer(self):
        with patch.object(cli, "demo") as demo:
            with patch.object(sys, "argv", ["lifting_lab", "demo", "--paused"]):
                with self.assertRaisesRegex(SystemExit, "requires --viewer"):
                    cli.main()
            demo.assert_not_called()

    def test_curriculum_cannot_be_combined_with_either_mass_override(self):
        for option in ("--mass", "--mass-fraction"):
            with self.subTest(option=option), patch.object(cli, "train") as train:
                with patch.object(sys, "argv", ["lifting_lab", "train", "--curriculum", option, "0"]):
                    with self.assertRaisesRegex(SystemExit, "curriculum selects masses"):
                        cli.main()
                train.assert_not_called()

    def test_evaluation_mass_cannot_override_protocol(self):
        for option in ("--mass", "--mass-fraction"):
            with self.subTest(option=option), patch.object(cli, "evaluate") as evaluate:
                with patch.object(sys, "argv", ["lifting_lab", "evaluate", option, "0"]):
                    with self.assertRaisesRegex(SystemExit, "Evaluation masses come from --protocol"):
                        cli.main()
                evaluate.assert_not_called()

    def test_valid_training_dispatch_passes_parsed_settings_without_running_training(self):
        arguments = ["lifting_lab", "train", "--device", "cpu", "--iterations", "1",
                     "--num-envs", "2", "--mass-fraction", ".025"]
        with patch.object(sys, "argv", arguments), patch.object(cli, "train") as train:
            with patch.dict(cli.os.environ):
                cli.main()
        train.assert_called_once()
        args = train.call_args.args[0]
        self.assertEqual((args.device, args.iterations, args.num_envs, args.mass_fraction),
                         ("cpu", 1, 2, .025))


class ComparisonScriptTests(unittest.TestCase):
    @staticmethod
    def record():
        return {
            "protocol_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "mass_observed": True,
            "support_mode": "forearms",
            "task_schema": "forearms-contact-v1",
            "observation_schema": "k1-load-state-97-v1",
            "observation_variant": "conditioned",
            "quick_smoke_only": False,
            "episode_seconds": 20.0,
            "episodes": 150,
            "batch_num_envs": 25,
            "training_seed": 11,
            "evaluation_seeds_used": [1011],
            "runtime": {
                "source_sha256": "3" * 64,
                "versions": {"mjlab": "1.6.0", "mujoco": "3.11.0"},
                "device": "cuda:0",
                "gpu": "fixture GPU A (no GPU is used)",
            },
            "member": "member1",
            "success_rate": .8,
        }

    def compare(self, reference, other):
        with TemporaryDirectory(prefix="lifting-lab-comparison-") as temporary:
            paths = [Path(temporary) / "member1.json", Path(temporary) / "member2.json"]
            for path, data in zip(paths, (reference, other)):
                path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts" / "compare_results.py"),
                 *(str(path) for path in paths)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                timeout=15, check=False,
            )
        self.assertEqual(result.stderr, "", result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_same_protocol_checkpoint_and_runtime_are_comparable(self):
        first = self.record()
        second = deepcopy(first)
        # A person's label, GPU model, or numerical outcome is not a config mismatch.
        second.update(member="member2", success_rate=.78)
        second["runtime"]["gpu"] = "fixture GPU B (no GPU is used)"
        code, result = self.compare(first, second)
        self.assertEqual(code, 0)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["mismatches"], [])
        self.assertEqual([run["member"] for run in result["runs"]], ["member1", "member2"])

    def test_different_protocol_is_rejected_with_exit_one(self):
        first = self.record()
        second = deepcopy(first)
        second["protocol_sha256"] = "4" * 64
        code, result = self.compare(first, second)
        self.assertEqual(code, 1)
        self.assertFalse(result["comparable"])
        self.assertEqual(len(result["mismatches"]), 1)
        self.assertIn("protocol_sha256 differs", result["mismatches"][0])

    def test_different_checkpoint_is_rejected_with_exit_one(self):
        first = self.record()
        second = deepcopy(first)
        second["checkpoint_sha256"] = "5" * 64
        code, result = self.compare(first, second)
        self.assertEqual(code, 1)
        self.assertFalse(result["comparable"])
        self.assertEqual(len(result["mismatches"]), 1)
        self.assertIn("checkpoint_sha256 differs", result["mismatches"][0])

    def test_different_mode_or_task_or_observation_schema_is_rejected(self):
        for key in ("support_mode", "task_schema", "observation_schema", "observation_variant"):
            with self.subTest(key=key):
                first, second = self.record(), self.record()
                second[key] = "different"
                code, result = self.compare(first, second)
                self.assertEqual(code, 1)
                self.assertIn(f"{key} differs", result["mismatches"][0])

    def test_missing_mode_metadata_is_not_silently_comparable(self):
        first, second = self.record(), self.record()
        del first["task_schema"]
        del second["task_schema"]
        code, result = self.compare(first, second)
        self.assertEqual(code, 1)
        self.assertFalse(result["comparable"])


if __name__ == "__main__":
    unittest.main()
