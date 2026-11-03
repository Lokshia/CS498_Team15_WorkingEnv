from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from team15_hardware.config import ConfigurationError, load_config, validate_config
from fixtures import make_config

CONTRACT = json.loads((ROOT.parent / "team15_proprio_package/tasks/team15_proprio/models/proprio_hold.contract.json").read_text())


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = make_config(CONTRACT)

    def invalid(self, profile, fragment):
        with self.assertRaises(ConfigurationError) as cm:
            validate_config(profile, CONTRACT)
        self.assertIn(fragment, str(cm.exception))
        return cm.exception

    def test_complete_synthetic_profile_valid_and_defensive_copy(self):
        checked = validate_config(self.profile, CONTRACT)
        self.assertEqual(checked["deployment_target"], "simulation_only")
        checked["joints"][0]["kp"] = 50
        self.assertEqual(self.profile["joints"][0]["kp"], 10)

    def test_template_reports_all_missing_site_values(self):
        data = json.loads((ROOT / "profiles/site.template.json").read_text())
        error = self.invalid(data, "review.robot_serial")
        for field in ("review.physical_support_and_estop_ready", "joints[0].kp",
                      "joints[21].tracking_error_limit_rad", "transport.motor_weight[21]",
                      "imu.body_from_imu[2][2]", "prepare.waypoints[0].duration_s",
                      "stop.procedure", "limits.status_rpc_timeout_s"):
            self.assertIn(field, str(error))

    def test_missing_sections_list_required_paths(self):
        error = self.invalid({}, "transport.low_state_topic")
        self.assertIn("joints[21].serial_index", str(error))
        self.assertIn("prepare.start_model_q", str(error))

    def test_wrong_protocol_model_or_target_rejected(self):
        cases = [(["protocol", "change_mode_api"], 1),
                 (["model", "onnx_sha256"], "bad"),
                 (["model", "control_dt_s"], .01),
                 (["deployment_target"], []),
                 (["upstream_booster_deploy_commit"], "main")]
        for path, value in cases:
            with self.subTest(path=path):
                p = deepcopy(self.profile)
                target = p
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.invalid(p, ".".join(path))

    def test_unreviewed_and_placeholder_evidence_rejected(self):
        self.profile["review"]["calibration_verified"] = False
        self.profile["review"]["evidence"] = "TODO"
        error = self.invalid(self.profile, "review.calibration_verified")
        self.assertIn("review.evidence", str(error))

    def test_index_permutation_name_and_sign(self):
        for key, value, message in (("serial_index", 1, "permutation"),
                                    ("serial_index", True, "serial_index"),
                                    ("name", "unexpected", "name"),
                                    ("sign", 0, "sign")):
            with self.subTest(key=key, value=value):
                p = deepcopy(self.profile)
                p["joints"][0][key] = value
                self.invalid(p, message)

    def test_nonfinite_string_boolean_numbers_rejected(self):
        for value in (float("nan"), float("inf"), "1", True, 10**500):
            with self.subTest(value=repr(value)[:30]):
                p = deepcopy(self.profile)
                p["joints"][0]["kp"] = value
                self.invalid(p, "joints[0].kp")

    def test_nonpositive_limits_and_bounds_rejected(self):
        for key, value in (("state_timeout_s", .21), ("command_timeout_s", .01),
                           ("inference_timeout_s", .02), ("max_trunk_tilt_deg", 45),
                           ("run_timeout_s", 0), ("status_rpc_timeout_s", 1)):
            with self.subTest(key=key):
                p = deepcopy(self.profile)
                p["limits"][key] = value
                self.invalid(p, key)

    def test_rotation_must_be_proper_so3(self):
        for matrix in ([[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                       [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [None, [0, 1, 0], [0, 0, 1]]):
            with self.subTest(matrix=matrix):
                p = deepcopy(self.profile)
                p["imu"]["body_from_imu"] = matrix
                self.invalid(p, "imu.body_from_imu")

    def test_nominal_and_start_must_fit_ranges(self):
        p = deepcopy(self.profile)
        p["joints"][0]["model_min_rad"] = .01
        self.invalid(p, "nominal pose")
        self.profile["prepare"]["start_model_q"][0] = 4.
        self.invalid(self.profile, "starting pose")

    def test_prepare_final_pose_and_speed_are_checked(self):
        p = deepcopy(self.profile)
        p["prepare"]["waypoints"][0]["model_q"][0] = .1
        self.invalid(p, "final pose")
        self.profile["prepare"]["start_model_q"][0] = 2.
        self.invalid(self.profile, "smoothstep peak target speed")

    def test_prepare_duration_and_ready_fields_checked(self):
        self.profile["prepare"]["waypoints"][0]["duration_s"] = 4.
        self.profile["prepare"]["ready_dwell_s"] = None
        error = self.invalid(self.profile, "total duration")
        self.assertIn("ready_dwell_s", str(error))

    def test_stop_and_transport_rejected_when_invalid(self):
        self.profile["stop"]["exit_mode"] = {"anything": 0}
        self.profile["transport"]["motor_weight"][0] = 2
        self.profile["transport"]["low_state_topic"] = "relative"
        error = self.invalid(self.profile, "stop.exit_mode")
        self.assertIn("motor_weight", str(error))
        self.assertIn("low_state_topic", str(error))

    def test_loader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.json"
            path.write_text('{"schema":"one","schema":"two"}')
            with self.assertRaisesRegex(ConfigurationError, "duplicate key"):
                load_config(path, CONTRACT)

    def test_loader_valid_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.json"
            path.write_text(json.dumps(self.profile))
            self.assertEqual(load_config(path, CONTRACT), self.profile)


if __name__ == "__main__":
    unittest.main()
