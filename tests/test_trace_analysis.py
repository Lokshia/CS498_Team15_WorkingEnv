"""NumPy/stdlib trace diagnostics tests; no simulation, torch, or GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_hold_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_hold_trace", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture(n=100, dt=.02):
    row = {"valid_hold": 1, "payload_forearm_supported": 1,
           "feet_contact_count": 2, "support_gap_steps": 0,
           "payload_tilt_deg": 2., "height_error_m": -.01,
           "torque_rms_nm": 3., "base_height_m": .5, "payload_height_m": .6,
           "fell": 0, "nonfinite": 0, "support_lost": 0,
           "payload_disallowed_contact": 0}
    return {"frames": n, "simulation_seconds": n * dt,
            "trace": [dict(row) for _ in range(n)], "runtime": {"source_sha256": "fixture"}}


class SummaryTests(unittest.TestCase):
    def test_fractions_extrema_and_absence_are_explicit(self):
        doc = fixture(4)
        for row, valid, arms, feet, gap, tilt, error, torque in zip(
            doc["trace"], [1, 0, 1, 1], [1, 0, 0, 1], [2, 0, 0, 1],
            [0, 1, 2, 0], [0, 1, 2, 3], [-.01, .02, -.03, .01], [1, 2, 3, 4],
        ):
            row.update(valid_hold=valid, payload_forearm_supported=arms,
                       feet_contact_count=feet, support_gap_steps=gap,
                       payload_tilt_deg=tilt, height_error_m=error, torque_rms_nm=torque)
        result = analysis.analyze_document(doc)
        self.assertFalse(result["formal_evaluation"])
        self.assertFalse(result["valid_hold"]["all_recorded_steps_valid"])
        self.assertEqual(result["valid_hold"]["first_invalid_step_1_based"], 2)
        self.assertEqual(result["valid_hold"]["valid_step_fraction"], .75)
        self.assertEqual(result["contact"]["bilateral_forearm_support_fraction"], .5)
        self.assertEqual(result["contact"]["both_feet_contact_fraction"], .25)
        self.assertEqual(result["contact"]["neither_foot_contact_fraction"], .5)
        self.assertEqual(result["contact"]["max_support_gap_steps"], 2)
        self.assertEqual(result["contact"]["longest_no_feet_contact_steps"], 2)
        self.assertEqual(result["height_error_absmax_m"], .03)
        self.assertEqual(result["payload_tilt_deg"]["max"], 3)
        self.assertAlmostEqual(result["payload_tilt_deg"]["p95"], 2.85)
        self.assertEqual(result["torque_rms_nm"]["mean"], 2.5)
        self.assertIsNone(result["optional_diagnostics"]["action_delta_rms"])
        self.assertIsNone(result["optional_diagnostics"]["joint_velocity_rms_rad_s"])
        self.assertNotIn("success", result)
        json.dumps(result, allow_nan=False)

    def test_optional_present_and_future_scalar_diagnostic_are_aggregated(self):
        doc = fixture(4)
        for i, row in enumerate(doc["trace"]):
            row.update(joint_velocity_rms_rad_s=i, action_delta_rms=i * .1,
                       action_delta_max_abs=i * .2)
        result = analysis.analyze_document(doc)["optional_diagnostics"]
        self.assertEqual(result["joint_velocity_rms_rad_s"]["mean"], 1.5)
        self.assertAlmostEqual(result["action_delta_rms"]["p95"], .285)
        self.assertAlmostEqual(result["action_delta_max_abs"]["max"], .6)

    def test_sampling_interval_is_inferred_not_assumed_50hz(self):
        result = analysis.analyze_document(fixture(400, .01))
        self.assertEqual(result["dt_seconds"], .01)
        self.assertEqual(result["sampling_rate_hz"], 100)

    def test_recorded_valid_is_not_reinterpreted_as_complete_episode_success(self):
        result = analysis.analyze_document(fixture(10))
        self.assertTrue(result["valid_hold"]["all_recorded_steps_valid"])
        self.assertFalse(result["formal_evaluation"])
        self.assertEqual(result["simulation_seconds"], .2)

    def test_input_document_is_not_mutated(self):
        doc = fixture()
        original = deepcopy(doc)
        analysis.analyze_document(doc)
        self.assertEqual(doc, original)

    def test_longest_run_at_boundaries(self):
        self.assertEqual(analysis.longest_run([True, True, False, True, True, True]), 3)
        self.assertEqual(analysis.longest_run([]), 0)


class ValidationTests(unittest.TestCase):
    def assert_invalid(self, modify, pattern):
        doc = fixture()
        modify(doc)
        with self.assertRaisesRegex(ValueError, pattern):
            analysis.analyze_document(doc)

    def test_missing_required_field_in_later_row_fails(self):
        self.assert_invalid(lambda d: d["trace"][3].pop("payload_forearm_supported"), "missing required")

    def test_count_empty_and_nonobject_rows_fail(self):
        self.assert_invalid(lambda d: d.update(frames=99), "row count")
        self.assert_invalid(lambda d: d.update(trace=[]), "nonempty")
        self.assert_invalid(lambda d: d["trace"].__setitem__(3, None), "must be an object")
        self.assert_invalid(lambda d: d.update(frames=True), "integer")

    def test_nonpositive_and_nonfinite_duration_fail(self):
        for seconds in (0, -1, float("nan"), float("inf"), None, "2", True):
            with self.subTest(seconds=seconds):
                self.assert_invalid(lambda d: d.update(simulation_seconds=seconds), "simulation_seconds")

    def test_nonfinite_known_unknown_and_nested_values_fail(self):
        for field in ("base_height_m", "new_diagnostic"):
            self.assert_invalid(lambda d: d["trace"][2].update({field: float("nan")}), "nonfinite")
        self.assert_invalid(lambda d: d["trace"][2].update(extra=[1, float("inf")]), "nonfinite")

    def test_partial_optional_is_not_silently_averaged(self):
        self.assert_invalid(lambda d: d["trace"][0].update(action_delta_rms=.1), "only part")

    def test_null_optional_present_is_not_fabricated_as_zero(self):
        doc = fixture()
        for row in doc["trace"]:
            row["action_delta_rms"] = None
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            analysis.analyze_document(doc)

    def test_invalid_flags_counts_and_negative_rms_fail(self):
        for key, value, pattern in (("valid_hold", .5, "0/1"), ("feet_contact_count", 3, "0, 1, or 2"),
                                    ("support_gap_steps", 1.5, "integer counts"),
                                    ("torque_rms_nm", -1, "nonnegative")):
            with self.subTest(key=key):
                self.assert_invalid(lambda d: d["trace"][0].update({key: value}), pattern)


class SpectrumTests(unittest.TestCase):
    def test_sinusoid_with_linear_drift_recovers_frequency(self):
        t = np.arange(1000) * .02
        result = analysis.spectrum(.5 + .01 * t + .005 * np.sin(2 * np.pi * 8 * t), .02)
        self.assertAlmostEqual(result["peak_hz"], 8)
        self.assertAlmostEqual(result["resolution_hz"], .05)
        self.assertFalse(result["possible_drift"])
        self.assertGreater(result["peak_three_bin_power_fraction"], .99)

    def test_short_trace_has_no_frequency(self):
        for n, dt in ((150, .02), (50, .1)):
            with self.subTest(n=n):
                result = analysis.spectrum(np.sin(np.arange(n)), dt)
                self.assertIsNone(result["peak_hz"])
                self.assertEqual(result["status"], "short_trace")

    def test_constant_or_linear_trace_does_not_invent_oscillation(self):
        for values in (np.ones(500), np.arange(500) * .001 + .5):
            result = analysis.spectrum(values, .02)
            self.assertIsNone(result["peak_hz"])
            self.assertEqual(result["status"], "constant_or_linear_trend")

    def test_low_frequency_is_labelled_possible_drift(self):
        t = np.arange(1000) * .02
        result = analysis.spectrum(np.sin(2 * np.pi * .2 * t), .02)
        self.assertAlmostEqual(result["peak_hz"], .2)
        self.assertTrue(result["possible_drift"])


class FileAndCliTests(unittest.TestCase):
    def test_directory_discovery_duplicate_inputs_and_source_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a" / "demo.json", root / "b" / "demo.json"
            for path in (first, second):
                path.parent.mkdir()
                path.write_text(json.dumps(fixture()), encoding="utf-8")
            summary = analysis.analyze_paths([root, first])
            self.assertEqual(summary["trace_count"], 2)
            self.assertEqual([r["name"] for r in summary["traces"]], ["a", "b"])
            self.assertEqual(len(summary["traces"][0]["source_file_sha256"]), 64)

    def test_missing_or_empty_inputs_fail(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No demo.json"):
                analysis.collect_paths([directory])
            with self.assertRaisesRegex(ValueError, "does not exist"):
                analysis.collect_paths([Path(directory) / "missing"])

    def test_existing_output_is_rejected_before_reading_inputs(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            output.write_text("keep", encoding="utf-8")
            with patch.object(analysis, "analyze_paths") as analyze:
                with self.assertRaises(FileExistsError):
                    analysis.main(["missing.json", "--output", str(output)])
            analyze.assert_not_called()
            self.assertEqual(output.read_text(), "keep")

    def test_bad_input_does_not_create_output(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "demo.json", Path(directory) / "out.json"
            source.write_text('{"trace": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.main([str(source), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_cli_creates_new_summary_and_never_overwrites(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "demo.json"
            output = Path(directory) / "new" / "summary.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            command = [sys.executable, "-B", str(SCRIPT), str(source), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(first.returncode, 0, first.stderr)
            raw = output.read_bytes()
            result = json.loads(raw)
            self.assertEqual(result["trace_count"], 1)
            self.assertFalse(result["formal_evaluation"])
            second = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
