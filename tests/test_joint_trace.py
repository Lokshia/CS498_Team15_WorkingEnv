"""Pure CPU NumPy summary/shape tests. Never construct an environment or load torch."""
from contextlib import redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_joint_trace.py"
SPEC = importlib.util.spec_from_file_location("capture_joint_trace", SCRIPT)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def fixture(n=3):
    names = np.array([f"joint_{i:02d}" for i in range(22)])
    arrays = {key: np.zeros((n, 22)) for key in capture.ARRAY_FIELDS}
    arrays.update(joint_names=names, action_joint_names=names.copy(),
                  times=np.arange(1, n + 1) * .02)
    return arrays


class SummaryTests(unittest.TestCase):
    def test_known_velocity_torque_and_difference_statistics(self):
        data = fixture()
        data["joint_vel"][:, 0] = [3, -4, 0]
        data["qfrc_actuator"][:, 0] = [3, 4, -12]
        data["joint_pos_target"][:, 0] = [1, 3, 6]
        data["action"][:, 0] = [.2, .4, .1]
        result = capture.summarize_arrays(data)
        row = result["joints"][0]
        self.assertAlmostEqual(row["joint_velocity_rms_rad_s"], np.sqrt(25 / 3))
        self.assertEqual(row["joint_velocity_max_abs_rad_s"], 4)
        self.assertAlmostEqual(row["joint_pos_target_delta_rms_rad_per_step"], np.sqrt(6.5))
        self.assertAlmostEqual(row["action_delta_rms_per_step"], np.sqrt(.065))
        self.assertAlmostEqual(row["actuator_torque_rms_nm"], np.sqrt(169 / 3))
        self.assertEqual(row["actuator_torque_max_abs_nm"], 12)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["difference_samples"], 2)
        self.assertFalse(result["formal_evaluation"])
        self.assertNotIn("success", result)
        json.dumps(result, allow_nan=False)

    def test_action_order_is_mapped_by_name_not_assumed_joint_order(self):
        data = fixture()
        data["action_joint_names"] = data["joint_names"][::-1].copy()
        data["action"][:, 21] = [0, 1, 3]
        data["action"][:, 0] = [0, 0, 0]
        result = capture.summarize_arrays(data)
        self.assertEqual(result["joints"][0]["action_index"], 21)
        self.assertAlmostEqual(result["joints"][0]["action_delta_rms_per_step"], np.sqrt(2.5))
        self.assertEqual(result["joints"][21]["action_delta_rms_per_step"], 0)

    def test_single_sample_differences_are_null_not_zero(self):
        result = capture.summarize_arrays(fixture(1))
        self.assertEqual(result["difference_samples"], 0)
        self.assertIsNone(result["joints"][0]["action_delta_rms_per_step"])
        self.assertIsNone(result["joints"][0]["joint_pos_target_delta_rms_rad_per_step"])
        self.assertIsNone(result["action_delta_rms_over_all_samples_per_step"])
        self.assertFalse(result["formal_evaluation"])

    def test_summary_does_not_mutate_input(self):
        data = fixture()
        original = deepcopy(data)
        capture.summarize_arrays(data)
        for key in original:
            np.testing.assert_array_equal(data[key], original[key])

    def test_first_difference_excludes_reset_to_first_target(self):
        data = fixture(2)
        data["joint_pos_target"][:] = 2
        data["action"][:] = .5
        result = capture.summarize_arrays(data)
        self.assertEqual(result["joint_pos_target_delta_rms_over_all_samples_rad_per_step"], 0)
        self.assertEqual(result["action_delta_rms_over_all_samples_per_step"], 0)


class ShapeAndFiniteTests(unittest.TestCase):
    def test_each_joint_array_requires_frames_by_22(self):
        for key in capture.ARRAY_FIELDS:
            for shape in ((3, 21), (2, 22), (3, 1, 22), (22,)):
                with self.subTest(key=key, shape=shape):
                    data = fixture()
                    data[key] = np.zeros(shape)
                    with self.assertRaisesRegex(ValueError, "frames, 22"):
                        capture.validate_arrays(data)

    def test_missing_array_is_rejected(self):
        data = fixture()
        del data["joint_pos_target"]
        with self.assertRaisesRegex(ValueError, "Missing array"):
            capture.validate_arrays(data)

    def test_nonfinite_joint_and_time_values_are_rejected(self):
        for key in (*capture.ARRAY_FIELDS, "times"):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(key=key, bad=bad):
                    data = fixture()
                    data[key].flat[0] = bad
                    with self.assertRaisesRegex(ValueError, "finite"):
                        capture.validate_arrays(data)

    def test_empty_and_nonuniform_or_nonpositive_times_are_rejected(self):
        for times in ([], [0, .02, .04], [.02, .02, .04], [.02, .05, .07], [.02, .01, .06]):
            with self.subTest(times=times):
                data = fixture()
                data["times"] = np.asarray(times)
                with self.assertRaises(ValueError):
                    capture.validate_arrays(data)

    def test_object_numeric_arrays_are_rejected(self):
        data = fixture()
        data["action"] = data["action"].astype(object)
        with self.assertRaisesRegex(ValueError, "numeric"):
            capture.validate_arrays(data)

    def test_names_must_be_22_unique_nonempty_strings_with_exact_action_mapping(self):
        variants = [np.array(["same"] * 22), np.array([""] * 22), np.arange(22),
                    np.array([f"j{i}" for i in range(21)]), np.array([f"j{i}" for i in range(22)], dtype=object)]
        for names in variants:
            with self.subTest(dtype=names.dtype):
                data = fixture()
                data["joint_names"] = names
                with self.assertRaises(ValueError):
                    capture.validate_arrays(data)
        data = fixture()
        data["action_joint_names"][0] = "other"
        with self.assertRaisesRegex(ValueError, "map exactly"):
            capture.validate_arrays(data)

    def test_tensor_snapshot_copies_mutable_source(self):
        array = np.ones((1, 22))
        tensor = Mock()
        tensor.detach.return_value.cpu.return_value.numpy.return_value = array
        snapshot = capture._snapshot(tensor, "test")
        array[:] = 99
        np.testing.assert_array_equal(snapshot, np.ones(22))


class SerializationAndParserTests(unittest.TestCase):
    def test_npz_roundtrip_needs_no_pickle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "joint_trace.npz"
            data = fixture()
            capture.save_npz(path, data)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(data))
                for key in archive.files:
                    self.assertFalse(archive[key].dtype.hasobject)
                    np.testing.assert_array_equal(archive[key], data[key])
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                capture.save_npz(path, data)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_data_does_not_create_npz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            data = fixture()
            data["joint_vel"][0, 0] = np.nan
            with self.assertRaises(ValueError):
                capture.save_npz(path, data)
            self.assertFalse(path.exists())

    def test_json_writer_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            capture._write_new_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                capture._write_new_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_parser_defaults_match_requested_capture(self):
        args = capture.build_parser().parse_args(["--output", "new"])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.seed, 1011)
        self.assertEqual(args.seconds, 20)
        self.assertEqual(args.mass, .4949224923474894)
        self.assertIsNone(args.checkpoint)

    def test_parser_rejects_negative_or_nonfinite_duration_and_mass(self):
        for flag, values in (("--seconds", ("0", "-1", "nan", "inf")),
                             ("--mass", ("-1", "nan", "inf"))):
            for value in values:
                with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    capture.build_parser().parse_args(["--output", "new", flag, value])

    def test_demo_args_use_project_parser_without_enabling_viewer(self):
        args = capture.build_parser().parse_args(["--output", "new", "--checkpoint", "own.pt", "--mass-blind"])
        parser = Mock()
        cli_module = SimpleNamespace(build_parser=Mock(return_value=parser))
        result = capture.make_demo_args(args, cli_module)
        argv = parser.parse_args.call_args.args[0]
        self.assertEqual(argv[0], "demo")
        self.assertEqual(argv[argv.index("--seed") + 1], "1011")
        self.assertIn("--mass-blind", argv)
        self.assertNotIn("--viewer", argv)
        self.assertIs(result, parser.parse_args.return_value)


if __name__ == "__main__":
    unittest.main()
