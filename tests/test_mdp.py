"""CPU tensor tests for COM and explicitly approximate support diagnostics."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab.mdp import balance_reward, combined_com_w, com_margin_approx, foot_slip_cost
from lifting_lab import mdp


def synthetic_env(masses, positions, soles=None):
    env = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(body_mass=torch.tensor(masses, dtype=torch.float64)),
        data=SimpleNamespace(xipos=torch.tensor(positions, dtype=torch.float64)),
    ))
    env.num_envs = len(positions)
    if soles is not None:
        env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            site_pos_w=torch.tensor(soles, dtype=torch.float64))) }
        env._lifting_site_ids = ([], [], [0, 1])
    return env


class CenterOfMassTests(unittest.TestCase):
    def test_zero_mass_world_excluded_payload_included(self):
        env = synthetic_env([0.0, 3.0, 1.0],
                            [[[99, 99, 99], [0, 0, 0.5], [1, 0, 0.5]]])
        torch.testing.assert_close(combined_com_w(env),
                                   torch.tensor([[0.25, 0, 0.5]], dtype=torch.float64))

    def test_per_environment_masses_are_not_averaged_across_batch(self):
        env = synthetic_env([[0.0, 3.0, 1.0], [0.0, 3.0, 3.0]],
                            [[[99, 99, 99], [0, 0, 0.5], [1, 0, 0.5]]] * 2)
        torch.testing.assert_close(combined_com_w(env)[:, 0],
                                   torch.tensor([0.25, 0.5], dtype=torch.float64))

    def test_broadcast_model_mass_axis(self):
        env = synthetic_env([[0.0, 3.0, 1.0]],
                            [[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                             [[0, 0, 0], [10, 0, 0], [11, 0, 0]]])
        torch.testing.assert_close(combined_com_w(env)[:, 0],
                                   torch.tensor([0.25, 10.25], dtype=torch.float64))

    def test_uses_inertial_centers_not_body_frame_origins(self):
        env = synthetic_env([0.0, 2.0, 2.0],
                            [[[0, 0, 0], [0.1, 0, 0.4], [0.7, 0, 0.8]]])
        env.sim.data.xpos = torch.zeros_like(env.sim.data.xipos)
        torch.testing.assert_close(combined_com_w(env),
                                   torch.tensor([[0.4, 0, 0.6]], dtype=torch.float64))

    def test_increased_payload_mass_moves_com_towards_payload(self):
        positions = [[[0, 0, 0], [0, 0, 0.5], [0.3, 0, 0.6]]]
        light = combined_com_w(synthetic_env([0.0, 20.0, 0.5], positions))[0]
        heavy = combined_com_w(synthetic_env([0.0, 20.0, 2.0], positions))[0]
        self.assertGreater(heavy[0].item(), light[0].item())
        self.assertGreater(heavy[2].item(), light[2].item())
        self.assertLess(heavy[0].item(), 0.3)


class GeometricMarginTests(unittest.TestCase):
    def make_env(self, com_x, offset=(0.0, 0.0)):
        x, y = offset
        return synthetic_env([0.0, 1.0], [[[0, 0, 0], [com_x + x, y, 0.5]]],
                             [[[x, y - 0.1, 0], [x, y + 0.1, 0]]])

    def test_inside_boundary_and_outside_signed_margin(self):
        for com_x, expected in ((0.0, 0.07), (0.07, 0.0), (0.1, -0.03)):
            with self.subTest(com_x=com_x):
                self.assertAlmostEqual(com_margin_approx(self.make_env(com_x)).item(),
                                       expected, places=12)

    def test_translation_of_entire_environment_does_not_change_margin(self):
        original = com_margin_approx(self.make_env(0.03))
        moved = com_margin_approx(self.make_env(0.03, (5.0, -2.0)))
        torch.testing.assert_close(original, moved)

    def test_margin_is_geometry_only_not_an_airborne_stability_certificate(self):
        env = self.make_env(0.0)
        before = com_margin_approx(env)
        env.scene["robot"].data.site_pos_w[..., 2] = 1.0
        # This equality documents the limitation: a positive approximate margin
        # does not prove ground contact. Contact diagnostics are separate.
        torch.testing.assert_close(com_margin_approx(env), before)

    def test_airborne_feet_get_no_balance_reward_despite_positive_margin(self):
        env = self.make_env(0.0)
        contact = torch.zeros((1, 2, 1))
        env.scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=contact))
        self.assertGreater(com_margin_approx(env).item(), 0)
        self.assertEqual(balance_reward(env).item(), 0)
        contact[0, 0, 0] = 1
        self.assertEqual(balance_reward(env).item(), 1)

    def test_slip_cost_only_uses_contacting_feet_and_horizontal_velocity(self):
        env = self.make_env(0.0)
        env.scene["robot"].data.site_lin_vel_w = torch.tensor(
            [[[3.0, 4.0, 100.0], [100.0, 100.0, 0.0]]])
        contact = torch.tensor([[[1.0], [0.0]]])
        env.scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=contact))
        self.assertEqual(foot_slip_cost(env).item(), 25.0)
        contact.zero_()
        self.assertEqual(foot_slip_cost(env).item(), 0.0)


class ForearmSupportTests(unittest.TestCase):
    def make_env(self):
        # Distinct rows exercise per-environment history, no tensor-library mocks.
        def sensor(n):
            return SimpleNamespace(data=SimpleNamespace(
                found=torch.zeros(2, n), force=torch.zeros(2, n, 3),
                force_history=torch.zeros(2, n, 4, 3)))
        arms = sensor(2)
        arms.primary_names = ["right_hand_collision", "left_hand_collision"]
        return SimpleNamespace(num_envs=2, device="cpu", episode_length_buf=torch.zeros(2, dtype=torch.long),
            cfg=SimpleNamespace(support_grace_steps=6), scene={
                "forearm_payload_contact": arms,
                "forbidden_robot_payload_contact": sensor(3),
                "ground_payload_contact": sensor(1),
            })

    def test_bilateral_actual_upward_support_not_distance_or_force_norm(self):
        env = self.make_env()
        data = env.scene["forearm_payload_contact"].data
        data.found[:] = 1
        data.force[0, :, 2] = torch.tensor([2., 3.])
        data.force[1, 0, 0] = 50  # Lateral contact does not lift the box.
        data.force[1, 1, 2] = 3
        values = mdp.forearm_contact_metrics(env)
        self.assertEqual(values["forearm_contact_count"].tolist(), [2, 2])
        self.assertEqual(values["payload_forearm_supported"].tolist(), [True, False])
        self.assertEqual(values["forearm_support_force_n"].tolist(), [5., 3.])
        data.force[0, 0, 2] = -1  # Net positive total is not enough either.
        self.assertFalse(mdp.forearm_contact_metrics(env)["payload_forearm_supported"][0])

    def test_zero_force_contact_and_one_arm_do_not_count_as_supported(self):
        env = self.make_env()
        data = env.scene["forearm_payload_contact"].data
        data.found[:] = torch.tensor([[1, 1], [1, 0]])
        data.force[1, 0, 2] = 1
        self.assertEqual(mdp.forearm_contact_metrics(env)["payload_forearm_supported"].tolist(), [False, False])

    def test_left_right_mapping_does_not_assume_regex_order(self):
        env = self.make_env()
        env.scene["forearm_payload_contact"].data.found[0, 1] = 1
        values = mdp.forearm_contact_metrics(env)
        self.assertTrue(values["left_forearm_contact"][0])
        self.assertFalse(values["right_forearm_contact"][0])

    def test_startup_and_later_loss_each_have_six_steps_not_twelve(self):
        env = self.make_env()
        for step in range(1, 8):
            env.episode_length_buf.fill_(step)
            self.assertEqual(mdp.support_lost(env).tolist(), [step > 6] * 2)
            # Read repeatedly and commit twice: no additional elapsed time.
            for _ in range(3):
                self.assertEqual(mdp.forearm_contact_metrics(env)["support_gap_steps"].tolist(), [step] * 2)
            mdp.update_support_state(env)
            mdp.update_support_state(env)
        data = env.scene["forearm_payload_contact"].data
        data.found[:] = 1
        data.force[:, :, 2] = 1
        env.episode_length_buf += 1
        mdp.update_support_state(env)
        self.assertEqual(mdp.forearm_contact_metrics(env)["support_gap_steps"].tolist(), [0, 0])
        data.found.zero_()
        for step in range(9, 16):
            env.episode_length_buf.fill_(step)
            self.assertEqual(mdp.support_lost(env).tolist(), [step > 14] * 2)
            mdp.update_support_state(env)

    def test_partial_reset_invalidates_only_that_rows_same_step_cache(self):
        env = self.make_env()
        env.episode_length_buf.fill_(1)
        mdp.update_support_state(env)
        mdp.reset_support_state(env, torch.tensor([0]))
        env.episode_length_buf[0] = 0
        self.assertEqual(mdp.forearm_contact_metrics(env)["support_gap_steps"].tolist(), [0, 1])
        env.episode_length_buf += 1
        mdp.update_support_state(env)
        self.assertEqual(mdp.forearm_contact_metrics(env)["support_gap_steps"].tolist(), [1, 2])

    def test_committed_metrics_match_rl_sample_after_sensor_changes(self):
        env = self.make_env()
        env.episode_length_buf.fill_(1)
        mdp.update_support_state(env)
        data = env.scene["forearm_payload_contact"].data
        data.found[:] = 1
        data.force[:, :, 2] = 1
        self.assertEqual(mdp.forearm_contact_metrics(env)["payload_forearm_supported"].tolist(), [False, False])
        env.episode_length_buf += 1
        self.assertEqual(mdp.forearm_contact_metrics(env)["payload_forearm_supported"].tolist(), [True, True])

    def test_floor_and_other_robot_contacts_are_immediately_forbidden(self):
        env = self.make_env()
        env.scene["ground_payload_contact"].data.found[0, 0] = 1
        # A transient torso/hand-ball contact from an earlier physics substep.
        env.scene["forbidden_robot_payload_contact"].data.force_history[1, 2, 3, 0] = .5
        self.assertEqual(mdp.payload_disallowed_contact(env).tolist(), [True, True])

    def test_reset_step_zero_does_not_consume_contact_grace(self):
        env = self.make_env()
        for _ in range(20):
            self.assertEqual(mdp.forearm_contact_metrics(env)["support_gap_steps"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
