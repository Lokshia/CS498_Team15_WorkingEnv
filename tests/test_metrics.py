"""Fast, CPU-only tests for the complete-episode success contract.

Run from lifting-lab with ``python -m unittest discover -s tests -v``.
These tests do not construct a simulator, allocate CUDA memory, or train a policy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab.metrics import EpisodeAccumulator, episode_success


def samples(tilt, height, **extra):
    return {
        "payload_tilt_deg": np.atleast_1d(tilt).astype(float),
        "height_error_m": np.atleast_1d(height).astype(float),
        **{key: np.atleast_1d(value) for key, value in extra.items()},
    }


class EpisodeSuccessTests(unittest.TestCase):
    def test_requires_complete_episode(self):
        result = episode_success([999, 1000, 1001], 1000, [False] * 3,
                                 [0.0] * 3, [0.0] * 3)
        np.testing.assert_array_equal(result, [False, True, True])

    def test_thresholds_are_strict(self):
        result = episode_success([1000] * 4, 1000, [False] * 4,
                                 [9.999, 10.0, 0.0, 0.0],
                                 [0.04999, 0.0, 0.05, 0.05001])
        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_one_fall_is_failure(self):
        self.assertFalse(bool(episode_success(1000, 1000, True, 0.0, 0.0)))

    def test_nonfinite_errors_cannot_pass(self):
        result = episode_success([1000] * 4, 1000, [False] * 4,
                                 [np.nan, np.inf, 0.0, 0.0],
                                 [0.0, 0.0, np.nan, np.inf])
        self.assertFalse(np.any(result))

    def test_custom_limits(self):
        result = episode_success([5, 5], 5, [False, False], [14.9, 15.0],
                                 [0.099, 0.0], tilt_limit_deg=15.0,
                                 height_limit_m=0.1)
        np.testing.assert_array_equal(result, [True, False])


class EpisodeAccumulatorTests(unittest.TestCase):
    def test_no_samples_are_not_success(self):
        result = EpisodeAccumulator(2, 1000).finish()
        self.assertEqual(result["steps"], [0, 0])
        self.assertEqual(result["success"], [False, False])
        self.assertEqual(result["success_rate"], 0.0)

    def test_short_run_cannot_pass_then_completion_can(self):
        accumulator = EpisodeAccumulator(1, 3)
        for _ in range(2):
            accumulator.update(samples(0.0, 0.0), [False])
        self.assertEqual(accumulator.finish()["success"], [False])
        accumulator.update(samples(0.0, 0.0), [False])
        self.assertEqual(accumulator.finish()["success"], [True])
        np.testing.assert_array_equal(accumulator.active, [False])

    def test_temporary_violation_is_not_hidden_by_average(self):
        accumulator = EpisodeAccumulator(2, 100)
        for step in range(100):
            accumulator.update(samples([10.01 if step == 50 else 0.0, 0.0],
                                       [0.0, 0.051 if step == 50 else 0.0]),
                               [False, False])
        result = accumulator.finish()
        self.assertEqual(result["success"], [False, False])
        self.assertEqual(result["steps"], [100, 100])
        self.assertAlmostEqual(result["max_payload_tilt_deg"][0], 10.01)
        self.assertAlmostEqual(result["max_abs_height_error_m"][1], 0.051)
        self.assertLess(result["means_until_completion_or_fall"]
                        ["payload_tilt_deg"][0], 1.0)

    def test_height_error_is_absolute(self):
        accumulator = EpisodeAccumulator(2, 1)
        accumulator.update(samples([0.0, 0.0], [-0.049, -0.05]), [False, False])
        result = accumulator.finish()
        self.assertEqual(result["success"], [True, False])
        np.testing.assert_allclose(result["max_abs_height_error_m"], [0.049, 0.05])

    def test_terminated_episode_cannot_be_replaced_by_auto_reset_success(self):
        accumulator = EpisodeAccumulator(2, 3)
        accumulator.update(samples([2.0, 1.0], [0.01, 0.0]), [True, False])
        for _ in range(5):
            accumulator.update(samples([0.0, 1.0], [0.0, 0.0]), [False, False])
        result = accumulator.finish()
        self.assertEqual(result["steps"], [1, 3])
        self.assertEqual(result["fallen"], [True, False])
        self.assertEqual(result["success"], [False, True])
        self.assertEqual(result["success_rate"], 0.5)
        np.testing.assert_allclose(result["max_payload_tilt_deg"], [2.0, 1.0])

    def test_nan_is_a_persistent_failure(self):
        accumulator = EpisodeAccumulator(2, 2)
        accumulator.update(samples([np.nan, 0.0], [0.0, np.inf]), [False, False])
        accumulator.update(samples([0.0, 0.0], [0.0, 0.0]), [False, False])
        result = accumulator.finish()
        self.assertEqual(result["steps"], [1, 1])
        self.assertEqual(result["fallen"], [True, True])
        self.assertEqual(result["numerical_failure"], [True, True])
        self.assertEqual(result["success"], [False, False])

    def test_nonfinite_statistics_are_strict_json_null_not_zero(self):
        accumulator = EpisodeAccumulator(2, 1)
        accumulator.update(samples([np.nan, 1.0], [0.01, -np.inf]), [False, False])
        result = accumulator.finish()
        encoded = json.dumps(result, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["max_payload_tilt_deg"], [None, 1.0])
        self.assertEqual(decoded["max_abs_height_error_m"], [0.01, None])
        self.assertEqual(decoded["means_until_completion_or_fall"]
                         ["payload_tilt_deg"], [None, 1.0])
        self.assertEqual(decoded["success"], [False, False])

    def test_any_batch_metric_nonfinite_fails_and_freezes_only_that_episode(self):
        accumulator = EpisodeAccumulator(3, 2)
        accumulator.update(samples([1.0] * 3, [0.01] * 3,
                                   torque_rms=[np.inf, 2.0, 3.0],
                                   grip_error_m=[0.0, np.nan, 0.0]), [False] * 3)
        accumulator.update(samples([0.0] * 3, [0.0] * 3,
                                   torque_rms=[0.0, 0.0, 5.0],
                                   grip_error_m=[0.0] * 3), [False] * 3)
        result = accumulator.finish()
        self.assertEqual(result["steps"], [1, 1, 2])
        self.assertEqual(result["fallen"], [True, True, False])
        self.assertEqual(result["numerical_failure"], [True, True, False])
        self.assertEqual(result["success"], [False, False, True])
        self.assertEqual(result["means_until_completion_or_fall"]["torque_rms"],
                         [None, 2.0, 4.0])
        json.dumps(result, allow_nan=False)

    def test_a_regular_termination_is_not_a_numerical_failure(self):
        accumulator = EpisodeAccumulator(1, 1)
        accumulator.update(samples(1.0, 0.01), [True])
        result = accumulator.finish()
        self.assertEqual(result["fallen"], [True])
        self.assertEqual(result["numerical_failure"], [False])
        self.assertEqual(result["success"], [False])

    def test_running_statistic_overflow_is_failed_and_json_safe(self):
        accumulator = EpisodeAccumulator(1, 3)
        for _ in range(3):
            accumulator.update(samples(1.0, 0.01, diagnostic=np.finfo(float).max),
                               [False])
        result = accumulator.finish()
        self.assertEqual(result["steps"], [2])
        self.assertEqual(result["numerical_failure"], [True])
        self.assertEqual(result["success"], [False])
        self.assertEqual(result["means_until_completion_or_fall"]["diagnostic"], [None])
        json.dumps(result, allow_nan=False)

    def test_non_batch_metadata_is_not_an_episode_metric(self):
        accumulator = EpisodeAccumulator(2, 1)
        accumulator.update(samples([1.0] * 2, [0.01] * 2,
                                   metadata=np.ones((2, 2)) * np.nan), [False] * 2)
        result = accumulator.finish()
        self.assertEqual(result["success"], [True, True])
        self.assertEqual(result["numerical_failure"], [False, False])
        self.assertNotIn("metadata", result["means_until_completion_or_fall"])

    def test_finished_episode_does_not_accept_later_samples(self):
        accumulator = EpisodeAccumulator(1, 2)
        accumulator.update(samples(1.0, 0.01), [False])
        accumulator.update(samples(3.0, -0.01), [False])
        before = accumulator.finish()
        accumulator.update(samples(np.nan, 100.0), [True])
        self.assertEqual(accumulator.finish(), before)
        self.assertAlmostEqual(before["means_until_completion_or_fall"]
                               ["payload_tilt_deg"][0], 2.0)

    def test_corrective_steps_are_not_automatic_failure(self):
        accumulator = EpisodeAccumulator(1, 2)
        for _ in range(2):
            accumulator.update(samples(2.0, 0.01, foot_contact_transitions=5,
                                       support_margin_m=-0.01), [False])
        self.assertEqual(accumulator.finish()["success"], [True])

    def test_finish_returns_snapshot_not_mutable_internal_arrays(self):
        accumulator = EpisodeAccumulator(1, 2)
        accumulator.update(samples(1.0, 0.01), [False])
        snapshot = accumulator.finish()
        snapshot["steps"][0] = 1000
        self.assertEqual(accumulator.finish()["steps"], [1])


class ForearmEpisodeTests(unittest.TestCase):
    def sample(self, supported=True, illegal=False, lost=False):
        return samples(0.0, 0.0, payload_forearm_supported=supported,
                       payload_disallowed_contact=illegal, support_lost=lost,
                       forearm_contact_count=2 if supported else 0,
                       forearm_support_force_n=1.0 if supported else 0.0)

    def test_six_startup_gap_steps_allowed_but_seventh_fails(self):
        acc = EpisodeAccumulator(1, 20, support_mode="forearms")
        for _ in range(6):
            acc.update(self.sample(False), [False])
        self.assertTrue(acc.active[0])
        acc.update(self.sample(False), [False])
        self.assertFalse(acc.active[0])
        self.assertEqual(acc.finish()["contact_failure"], [True])
        self.assertEqual(acc.finish()["steps"], [7])

    def test_contact_recovery_resets_gap_and_complete_episode_passes(self):
        acc = EpisodeAccumulator(1, 14, support_mode="forearms")
        for step in range(14):
            acc.update(self.sample(supported=step in (6, 13)), [False])
        self.assertEqual(acc.finish()["success"], [True])
        self.assertEqual(acc.finish()["max_contact_gap_steps"], [6])

    def test_illegal_contact_fails_immediately_even_during_startup_and_cannot_reset(self):
        acc = EpisodeAccumulator(1, 3, support_mode="forearms")
        acc.update(self.sample(illegal=True), [False])
        for _ in range(5):
            acc.update(self.sample(), [False])
        self.assertEqual(acc.finish()["success"], [False])
        self.assertEqual(acc.finish()["disallowed_support"], [True])
        self.assertEqual(acc.finish()["steps"], [1])

    def test_authoritative_per_arm_support_not_inferred_from_total_force(self):
        acc = EpisodeAccumulator(1, 10, support_mode="forearms", contact_grace_steps=0)
        metric = self.sample(supported=False)
        metric["forearm_contact_count"][:] = 2
        metric["forearm_support_force_n"][:] = 100
        acc.update(metric, [False])
        self.assertEqual(acc.finish()["contact_failure"], [True])

    def test_missing_contact_evidence_is_an_error(self):
        acc = EpisodeAccumulator(1, 10, support_mode="forearms")
        with self.assertRaises(KeyError):
            acc.update(samples(0.0, 0.0), [False])

    def test_premature_timeout_cannot_reset_into_success(self):
        acc = EpisodeAccumulator(1, 3)
        acc.update(samples(0.0, 0.0), [False], [True])
        for _ in range(5):
            acc.update(samples(0.0, 0.0), [False], [False])
        self.assertEqual(acc.finish()["success"], [False])
        self.assertEqual(acc.finish()["early_truncation"], [True])
        self.assertEqual(acc.finish()["steps"], [1])

    def test_timeout_exactly_at_required_duration_is_not_failure(self):
        acc = EpisodeAccumulator(1, 1)
        acc.update(samples(0.0, 0.0), [False], [True])
        self.assertEqual(acc.finish()["success"], [True])

    def test_post_forward_fall_on_final_step_is_failure_without_terminated_flag(self):
        acc = EpisodeAccumulator(1, 1)
        acc.update(samples(0.0, 0.0, fell=True), [False])
        self.assertEqual(acc.finish()["success"], [False])
        self.assertEqual(acc.finish()["fallen"], [True])

    def test_post_forward_nonfinite_flag_is_failure_even_with_finite_metric_values(self):
        acc = EpisodeAccumulator(1, 1)
        acc.update(samples(0.0, 0.0, nonfinite=True), [False])
        self.assertEqual(acc.finish()["success"], [False])
        self.assertEqual(acc.finish()["numerical_failure"], [True])


if __name__ == "__main__":
    unittest.main()
