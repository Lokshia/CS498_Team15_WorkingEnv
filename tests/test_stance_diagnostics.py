"""Independent numerical checks for stance measurement geometry and statistics."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("diagnose_stance", SCRIPTS / "diagnose_stance.py")
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)
sys.path.pop(0)


class StanceGeometryTests(unittest.TestCase):
    def test_leg_mirror_uses_names_and_signs_with_reordered_columns(self):
        pairs = ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
        names = [f"{side}_{name}" for side in ("Left", "Right") for name in pairs]
        positions = np.array([[.2, .1, .3, .8, -.4, -.1, .2, -.1, -.3, .8, -.4, .1]])
        # Only the knee is deliberately asymmetric by ten degrees.
        positions[0, names.index("Left_Knee_Pitch")] += np.radians(10)
        order = np.arange(len(names))[::-1]
        data = {"joint_names": np.array(names)[order], "joint_pos_rad": positions[:, order]}
        result = diagnose.summarize_leg_mirror(data, np.array([True]))
        self.assertAlmostEqual(result["rms_deg_over_frames_and_pairs"], 10 / np.sqrt(6))
        self.assertAlmostEqual(result["max_abs_deg_over_frames_and_pairs"], 10)
        np.testing.assert_allclose(result["rms_deg_by_pair"], [0, 0, 0, 10, 0, 0], atol=1e-10)
        self.assertEqual(result["all_pairs_within_5deg_fraction"], 0)

    def test_known_roll_pitch_and_yaw_do_not_confuse_tilt(self):
        half = np.pi / 12
        q = np.array([[1, 0, 0, 0], [np.cos(half), np.sin(half), 0, 0],
                      [np.cos(half), 0, np.sin(half), 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(diagnose.sole_tilt_degrees(q), [0, 30, 30, 0], atol=1e-6)

    def test_upside_down_is_not_reported_flat(self):
        np.testing.assert_allclose(diagnose.sole_tilt_degrees([[0, 1, 0, 0]]), [180])

    def test_quaternion_scale_and_sign_do_not_change_tilt(self):
        q = np.array([.9, .1, .2, .3])
        self.assertAlmostEqual(float(diagnose.sole_tilt_degrees(q)), float(diagnose.sole_tilt_degrees(-q * 2)))
        with self.assertRaises(ValueError):
            diagnose.sole_tilt_degrees([0, 0, 0, 0])

    def test_load_ratio_excludes_airborne_samples_and_flatness_requires_contact(self):
        force = np.zeros((3, 2, 3))
        force[:, :, 2] = [[100, 100], [160, 40], [0, 0]]
        data = {"time_s": np.array([.02, .04, .06]), "ground_reaction_world_n": force,
                "foot_contact_count": np.array([[1, 1], [1, 1], [0, 0]]),
                "sole_tilt_deg": np.array([[0, 0], [15, 0], [0, 0]]),
                "joint_torque_nm": np.ones((3, 22)) * 3,
                "joint_vel_rad_s": np.zeros((3, 22)), "robot_tilt_deg": np.zeros(3)}
        result = diagnose.summarize_window(data, np.ones(3, dtype=bool))
        self.assertEqual(result["loaded_frames"], 2)
        self.assertAlmostEqual(result["left_load_fraction_mean_when_loaded"], .65)
        self.assertAlmostEqual(result["both_feet_contact_fraction"], 2 / 3)
        self.assertAlmostEqual(result["both_feet_contact_and_tilt_under_5deg_fraction"], 1 / 3)
        self.assertAlmostEqual(result["joint_torque_rms_nm"], 3)
        self.assertIsNone(diagnose.summarize_window(data, np.zeros(3, dtype=bool)))


if __name__ == "__main__":
    unittest.main()
