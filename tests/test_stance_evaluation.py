"""Numerical checks for complete-episode stance scoring and failure freezing."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_stance import strict_stance_success, summarize_episode, condition_summary
from lifting_lab.metrics import EpisodeAccumulator


class StanceEvaluationTests(unittest.TestCase):
    def test_strict_success_requires_complete_hold_and_steady_window(self):
        good = {"both_feet_contact_and_tilt_under_5deg_fraction": .95}
        self.assertTrue(strict_stance_success(True, good, 901, 901))
        self.assertFalse(strict_stance_success(False, good, 901, 901))
        self.assertFalse(strict_stance_success(True, good, 900, 901))
        self.assertFalse(strict_stance_success(True, None, 0, 0))
        self.assertFalse(strict_stance_success(True,
            {"both_feet_contact_and_tilt_under_5deg_fraction": .949}, 901, 901))

    def test_failed_slot_reset_samples_never_improve_episode(self):
        count, batch = 4, 2
        data = {
            "time_s": np.arange(1, count + 1, dtype=float),
            "ground_reaction_world_n": np.tile([0., 0., 100.], (count, batch, 2, 1)),
            "foot_contact_count": np.ones((count, batch, 2)),
            "sole_tilt_deg": np.zeros((count, batch, 2)),
            "joint_torque_nm": np.ones((count, batch, 22)),
            "joint_vel_rad_s": np.ones((count, batch, 22)),
            "robot_tilt_deg": np.zeros((count, batch)),
        }
        # Slot 1 fails at step 2; any state after that is a replacement reset.
        accumulator = EpisodeAccumulator(batch, count, support_mode="forearms")
        active = []
        for step in range(count):
            active.append(accumulator.active.copy())
            # Later zero termination flags represent the reset slot continuing
            # alongside the surviving original episode.
            metrics = {
                "payload_tilt_deg": np.zeros(batch), "height_error_m": np.zeros(batch),
                "payload_disallowed_contact": np.array([False, step == 1]),
                "support_lost": np.zeros(batch, dtype=bool),
                "payload_forearm_supported": np.ones(batch, dtype=bool),
            }
            accumulator.update(metrics, [False, step == 1], [False, False])
        active = np.asarray(active)
        np.testing.assert_array_equal(active, [[1, 1], [1, 1], [1, 0], [1, 0]])
        data["sole_tilt_deg"][:2, 1] = 20
        data["ground_reaction_world_n"][2:, 1] = 1e6
        data["joint_torque_nm"][2:, 1] = 1e6
        hold = accumulator.finish()
        records = [summarize_episode(data, active, i, hold, count, 1.) for i in range(batch)]
        self.assertTrue(records[0]["strict_stance_success"])
        self.assertFalse(records[1]["strict_stance_success"])
        self.assertEqual(records[1]["steps"], 2)
        self.assertEqual(records[1]["steady_from_2s"]["frames"], 1)
        self.assertEqual(records[1]["all"]["vertical_force_mean_n_left_right"], [100., 100.])
        self.assertEqual(records[1]["all"]["joint_torque_rms_nm"], 1.)
        self.assertEqual(records[1]["steady_from_2s"]["both_feet_contact_and_tilt_under_5deg_fraction"], 0.)
        summary = condition_summary(records)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["hold_success_rate"], .5)
        self.assertEqual(summary["strict_stance_success_rate"], .5)
        self.assertEqual(summary["steady_means_hold_successful_episodes"]["episodes_with_steady_samples"], 1)


if __name__ == "__main__":
    unittest.main()
