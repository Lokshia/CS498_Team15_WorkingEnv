"""CPU tensor/configuration checks for the opt-in forearm reward ablation."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import mdp as builtin
from lifting_lab import mdp
from lifting_lab.env_cfg import make_env_cfg


BASELINE = {
    "hold_height": (mdp.hold_height_reward, 3.0),
    "payload_level": (mdp.payload_level_reward, 2.0),
    "upright": (mdp.upright_reward, 2.0),
    "balance_approx": (mdp.balance_reward, 0.5),
    "effort": (mdp.effort_cost, -0.00002),
    "smooth_actions": (builtin.action_rate_l2, -0.02),
    "payload_motion": (mdp.payload_motion_cost, -0.05),
    "foot_slip": (mdp.foot_slip_cost, -0.2),
    "joint_limits": (builtin.joint_pos_limits, -0.1),
    "supported_hold": (mdp.supported_hold_reward, 3.0),
}
NEW_COSTS = (mdp.unsupported_cost, mdp.missing_feet_cost,
             mdp.robot_vertical_velocity_cost, mdp.robot_angular_velocity_cost,
             mdp.joint_velocity_cost)
DIAGNOSTICS = ("joint_velocity_rms_rad_s", "action_delta_rms", "robot_vertical_speed_m_s",
               "robot_angular_speed_rad_s", "payload_linear_speed_m_s",
               "payload_angular_speed_rad_s", "both_feet_contact", "airborne")


def tensor_env(batch):
    """Small explicit state fixture; every reported field contains real tensors."""
    class Scene(dict):
        pass

    def state(sites):
        return SimpleNamespace(
            root_link_pos_w=torch.tensor([[0., 0., .6]]).repeat(batch, 1),
            root_link_lin_vel_w=torch.zeros(batch, 3), root_link_ang_vel_w=torch.zeros(batch, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(batch, 1),
            site_pos_w=torch.zeros(batch, sites, 3), site_lin_vel_w=torch.zeros(batch, sites, 3),
            joint_vel=torch.zeros(batch, 22), qfrc_actuator=torch.zeros(batch, 22))

    def sensor(count):
        return SimpleNamespace(data=SimpleNamespace(
            found=torch.zeros(batch, count), force=torch.zeros(batch, count, 3),
            force_history=None))

    scene = Scene(robot=SimpleNamespace(data=state(4)), payload=SimpleNamespace(data=state(2)))
    scene.env_origins = torch.zeros(batch, 3)
    scene["robot"].data.site_pos_w[:, 2:, 1] = torch.tensor([-.1, .1])
    for key, count in (("feet_ground_contact", 2), ("forearm_payload_contact", 2),
                       ("forbidden_robot_payload_contact", 1), ("ground_payload_contact", 1)):
        scene[key] = sensor(count)
    scene["forearm_payload_contact"].primary_names = ["left_hand_collision", "right_hand_collision"]
    return SimpleNamespace(num_envs=batch, device="cpu", scene=scene,
        cfg=SimpleNamespace(support_mode="forearms", support_grace_steps=6, reference_height_m=.6),
        episode_length_buf=torch.ones(batch, dtype=torch.long),
        _lifting_site_ids=([0, 1], [0, 1], [2, 3]),
        action_manager=SimpleNamespace(action=torch.zeros(batch, 22), prev_action=torch.zeros(batch, 22)),
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.tensor([[0., 20., 1.]])),
            data=SimpleNamespace(xipos=torch.zeros(batch, 3, 3), qpos=torch.zeros(batch, 30),
                                 qvel=torch.zeros(batch, 28))))


class RewardProfileTests(unittest.TestCase):
    def test_baseline_functions_weights_and_parameters_are_unchanged(self):
        for mode in ("pregrasp", "forearms"):
            expected = {key: value for key, value in BASELINE.items()
                        if mode == "forearms" or key != "supported_hold"}
            for options in ({}, {"reward_profile": "baseline"}):
                cfg = make_env_cfg(support_mode=mode, **options)
                self.assertEqual(cfg.reward_profile, "baseline")
                self.assertEqual(set(cfg.rewards), set(expected))
                for key, (func, weight) in expected.items():
                    self.assertIs(cfg.rewards[key].func, func)
                    self.assertEqual(cfg.rewards[key].weight, weight)
                    self.assertEqual(cfg.rewards[key].params, {})

    def test_smooth_profile_changes_only_requested_reward_terms(self):
        baseline = make_env_cfg(support_mode="forearms")
        smooth = make_env_cfg(support_mode="forearms", reward_profile="smooth-v1")
        expected = {key: weight for key, (_, weight) in BASELINE.items()}
        expected.update(smooth_actions=-.15, payload_motion=-.2, foot_slip=-.5,
                        unsupported=-2., missing_feet=-2., robot_vertical_velocity=-4.,
                        robot_angular_velocity=-.2, joint_velocity=-.002)
        self.assertEqual({key: term.weight for key, term in smooth.rewards.items()}, expected)
        for key, (func, _) in BASELINE.items():
            self.assertIs(smooth.rewards[key].func, func)
        for key in ("actions", "observations", "terminations", "events", "sim",
                    "decimation", "support_grace_steps", "episode_length_s"):
            self.assertEqual(getattr(baseline, key), getattr(smooth, key), key)
        self.assertEqual(baseline.scene.sensors, smooth.scene.sensors)
        self.assertEqual(baseline.rewards["smooth_actions"].weight, -.02)

    def test_rejects_unknown_profile_and_smooth_pregrasp(self):
        for kwargs in ({"reward_profile": "smooth"}, {"reward_profile": None},
                       {"reward_profile": "smooth-v1"},
                       {"reward_profile": "smooth-v1", "support_mode": "pregrasp"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_env_cfg(**kwargs)


class SmoothCostTests(unittest.TestCase):
    def test_contact_costs_distinguish_missing_feet_and_bilateral_support(self):
        env = tensor_env(3)
        env.scene["feet_ground_contact"].data.found[:] = torch.tensor([[1, 1], [1, 0], [0, 0]])
        arms = env.scene["forearm_payload_contact"].data
        arms.found[:] = 1
        arms.force[:, :, 2] = torch.tensor([[1., 1.], [1., 0.], [-1., 2.]])
        torch.testing.assert_close(mdp.missing_feet_cost(env), torch.tensor([0., 1., 2.]))
        torch.testing.assert_close(mdp.unsupported_cost(env), torch.tensor([0., 1., 1.]))
        # A grace period prevents termination, but never exempts this cost.
        self.assertFalse(mdp.support_lost(env).any())

    def test_motion_costs_are_unclipped_world_velocity_squares(self):
        env = tensor_env(2)
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_w[:] = torch.tensor([[100., 100., -3.], [0., 0., 2.]])
        robot.root_link_ang_vel_w[:] = torch.tensor([[1., 2., 3.], [0., 0., 0.]])
        robot.joint_vel[0, :2] = torch.tensor([3., 4.])
        for func, expected in ((mdp.robot_vertical_velocity_cost, [9., 4.]),
                               (mdp.robot_angular_velocity_cost, [14., 0.]),
                               (mdp.joint_velocity_cost, [25., 0.])):
            torch.testing.assert_close(func(env), torch.tensor(expected))

    def test_single_and_batched_values_are_finite_and_rows_are_independent(self):
        for batch in (1, 8):
            env = tensor_env(batch)
            for func in NEW_COSTS:
                value = func(env)
                self.assertEqual(value.shape, (batch,))
                self.assertTrue(torch.isfinite(value).all())
            env.scene["robot"].data.joint_vel[-1] = 2.
            expected = torch.zeros(batch)
            expected[-1] = 88.
            torch.testing.assert_close(mdp.joint_velocity_cost(env), expected)
            metrics = mdp.task_metrics(env)
            for key in DIAGNOSTICS:
                self.assertEqual(metrics[key].shape, (batch,), key)
                self.assertTrue(torch.isfinite(metrics[key]).all(), key)

    def test_diagnostics_units_and_no_change_to_valid_hold(self):
        env = tensor_env(3)
        robot = env.scene["robot"].data
        payload = env.scene["payload"].data
        robot.joint_vel[:] = 2.
        robot.root_link_lin_vel_w[:, 2] = -3.
        robot.root_link_ang_vel_w[:] = torch.tensor([3., 4., 0.])
        payload.root_link_lin_vel_w[:] = torch.tensor([0., 3., 4.])
        payload.root_link_ang_vel_w[:] = torch.tensor([0., 0., -2.])
        env.action_manager.action[:] = .6
        env.action_manager.prev_action[:] = .1
        env.scene["feet_ground_contact"].data.found[:] = torch.tensor([[1, 1], [1, 0], [0, 0]])
        metrics = mdp.task_metrics(env)
        for key, value in zip(DIAGNOSTICS[:6], (2., .5, 3., 5., 5., 2.), strict=True):
            torch.testing.assert_close(metrics[key], torch.full((3,), value))
        torch.testing.assert_close(metrics["both_feet_contact"], torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(metrics["airborne"], torch.tensor([0., 0., 1.]))
        # Diagnostics/reward costs are not new success or termination thresholds.
        torch.testing.assert_close(metrics["valid_hold"], torch.ones(3))


if __name__ == "__main__":
    unittest.main()
