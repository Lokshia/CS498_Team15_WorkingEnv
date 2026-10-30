"""Stance shaping must retain the policy/physics contract and use physical units."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab import cli, mdp
from lifting_lab.env_cfg import make_env_cfg


def fixture(batch=1, *, reversed_sensor=False):
    """Known states, independent of simulator allocation or contact solver."""
    names = ["Head_Yaw", "Head_Pitch"]
    for side in ("Left", "Right"):
        names += [f"{side}_{joint}" for joint in
                  ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")]
    for side in ("Left", "Right"):
        names += [f"{side}_{joint}" for joint in
                  ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]

    def find_joints(patterns, preserve_order=True):
        del preserve_order
        indices = [i for pattern in patterns for i, name in enumerate(names)
                   if re.fullmatch(pattern, name)]
        return indices, [names[i] for i in indices]

    nominal = torch.zeros(batch, 22)
    nominal[:, [10, 14, 16, 20]] = -.4
    nominal[:, [13, 19]] = .8
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(batch, 4, 1),
            joint_pos=nominal.clone(), default_joint_pos=nominal),
        joint_names=names, find_joints=find_joints)
    primary_names = ["robot/left_foot_link", "robot/right_foot_link"]
    if reversed_sensor:
        primary_names.reverse()
    contact = SimpleNamespace(primary_names=primary_names, data=SimpleNamespace(
        found=torch.ones(batch, 2),
        force=torch.tensor([0., 0., -100.]).repeat(batch, 2, 1)))
    return SimpleNamespace(
        num_envs=batch,
        scene={"robot": robot, "feet_ground_contact": contact},
        _lifting_site_ids=([0, 1], [0, 1], [2, 3]),
        _lifting_payload_body_id=1,
        sim=SimpleNamespace(model=SimpleNamespace(
            body_mass=torch.tensor([[0., .5]]).repeat(batch, 1))))


def set_vertical_reactions(env, values):
    """Specify upward forces ON each foot; sensor reports the opposite sign."""
    data = env.scene["feet_ground_contact"].data
    data.force.zero_()
    data.force[..., 2] = -torch.as_tensor(values, dtype=data.force.dtype)


class StancePhysicsTests(unittest.TestCase):
    def test_named_order_and_force_sign_are_explicit(self):
        env = fixture(reversed_sensor=True)
        # Raw sensor order is right, left, with a nonzero horizontal component.
        raw = torch.tensor([[[3., -4., -60.], [-2., 5., -140.]]])
        env.scene["feet_ground_contact"].data.force[:] = raw
        env.scene["feet_ground_contact"].data.found[:] = torch.tensor([[0., 1.]])
        normal_z, reaction, contact = mdp.stance_foot_data(env)
        torch.testing.assert_close(normal_z, torch.ones(1, 2))
        torch.testing.assert_close(reaction, torch.tensor([[[2., -5., 140.], [-3., 4., 60.]]]))
        torch.testing.assert_close(contact, torch.tensor([[True, False]]))

    def test_unknown_or_duplicate_foot_names_fail_instead_of_guessing(self):
        for names in (["left_foot_link", "left_foot_link"],
                      ["left_foot_link", "some_other_link"]):
            env = fixture()
            env.scene["feet_ground_contact"].primary_names = names
            with self.assertRaises(ValueError):
                mdp.stance_foot_data(env)

    def test_known_roll_pitch_and_inversion_not_just_projected_xy(self):
        env = fixture(4)
        quat = env.scene["robot"].data.site_quat_w
        roll = math.radians(30)
        quat[1, 2] = torch.tensor([math.cos(roll / 2), math.sin(roll / 2), 0., 0.])
        pitch = math.radians(60)
        quat[2, 3] = torch.tensor([math.cos(pitch / 2), 0., math.sin(pitch / 2), 0.])
        quat[3, 2] = torch.tensor([0., 1., 0., 0.])  # Sole facing downward.
        normal_z, _, _ = mdp.stance_foot_data(env)
        expected = torch.tensor([[1., 1.], [math.cos(roll), 1.], [1., .5], [-1., 1.]])
        torch.testing.assert_close(normal_z, expected)
        torch.testing.assert_close(mdp.feet_flat_cost(env), torch.tensor(
            [0., 1 - math.cos(roll), .5, 2.]))

    def test_yaw_does_not_change_tilt_or_flatness_cost(self):
        env = fixture(2)
        pitch, yaw = math.radians(25), math.radians(110)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        quat = env.scene["robot"].data.site_quat_w
        quat[0, 2:] = torch.tensor([cp, 0., sp, 0.])
        # q_z(yaw) * q_y(pitch), calculated independently.
        quat[1, 2:] = torch.tensor([cy * cp, -sy * sp, cy * sp, sy * cp])
        cost = mdp.feet_flat_cost(env)
        torch.testing.assert_close(cost[0], cost[1])

    def test_load_deadband_allows_thirty_seventy_not_only_equal_load(self):
        env = fixture(4)
        set_vertical_reactions(env, [[100., 100.], [60., 140.], [140., 60.], [10., 190.]])
        cost = mdp.foot_load_share_cost(env)
        torch.testing.assert_close(cost[:3], torch.zeros(3))
        self.assertGreater(float(cost[3]), .6)

    def test_missing_airborne_tensile_or_tiny_forces_are_not_balanced(self):
        env = fixture(5)
        set_vertical_reactions(env, [[0., 200.], [0., 0.], [-100., -100.], [1., 1.], [100., 100.]])
        env.scene["feet_ground_contact"].data.found[4] = 0
        torch.testing.assert_close(mdp.foot_load_share_cost(env), torch.ones(5))

    def test_leg_preference_keeps_bent_knees_and_does_not_constrain_arms(self):
        env = fixture(3)
        data = env.scene["robot"].data
        data.joint_pos[1, :10] += 1.0
        data.joint_pos[2, 13] += .25
        cost = mdp.leg_posture_cost(env)
        torch.testing.assert_close(cost[:2], torch.zeros(2))
        self.assertAlmostEqual(float(cost[2]), 1 / 12, places=6)

    def test_leg_mapping_is_checked(self):
        env = fixture()
        env.scene["robot"].find_joints = lambda *args, **kwargs: (list(range(11)), [])
        with self.assertRaises(ValueError):
            mdp.leg_posture_cost(env)

    def test_batch_rows_and_single_env_are_independent_and_finite(self):
        for batch in (1, 8):
            env = fixture(batch)
            env.scene["robot"].data.joint_pos[-1, 19] += .5
            for function in (mdp.feet_flat_cost, mdp.foot_load_share_cost, mdp.leg_posture_cost):
                values = function(env)
                self.assertEqual(tuple(values.shape), (batch,))
                self.assertTrue(bool(torch.isfinite(values).all()))
            expected = torch.zeros(batch)
            expected[-1] = 4 / 12
            torch.testing.assert_close(mdp.leg_posture_cost(env), expected)


class StanceProfileTests(unittest.TestCase):
    def test_new_profile_keeps_physics_and_policy_contract(self):
        smooth = make_env_cfg(support_mode="forearms", reward_profile="smooth-v1")
        before_rewards = deepcopy(smooth.rewards)
        stance = make_env_cfg(support_mode="forearms", reward_profile="stance-v1")
        self.assertEqual(stance.reward_profile, "stance-v1")
        for key in ("actions", "observations", "terminations", "events", "sim",
                    "decimation", "support_grace_steps", "episode_length_s"):
            self.assertEqual(getattr(stance, key), getattr(smooth, key), key)
        self.assertEqual(stance.scene.sensors, smooth.scene.sensors)
        self.assertEqual(smooth.rewards, before_rewards)
        # Do not reward the old support rectangle that includes unloaded feet.
        self.assertTrue("balance_approx" not in stance.rewards
                        or stance.rewards["balance_approx"].weight == 0)
        # Existing holding and anti-shaking objectives remain active.
        for name in ("hold_height", "payload_level", "supported_hold", "unsupported",
                     "smooth_actions", "payload_motion", "missing_feet"):
            self.assertIn(name, stance.rewards)
            self.assertIs(stance.rewards[name].func, smooth.rewards[name].func)
            self.assertEqual(stance.rewards[name].weight > 0,
                             smooth.rewards[name].weight > 0)

    def test_stance_profile_rejects_legacy_connected_box_task(self):
        with self.assertRaises(ValueError):
            make_env_cfg(support_mode="pregrasp", reward_profile="stance-v1")
        args = cli.build_parser().parse_args(
            ["smoke", "--support-mode", "pregrasp", "--reward-profile", "stance-v1"])
        with self.assertRaises(ValueError):
            cli.reward_profile(args)

    def test_default_profile_is_not_silently_changed(self):
        for command in ("smoke", "demo", "train", "evaluate"):
            args = cli.build_parser().parse_args([command])
            self.assertEqual(cli.reward_profile(args), "baseline")
        for profile in ("baseline", "smooth-v1"):
            args = cli.build_parser().parse_args(["train", "--reward-profile", profile])
            self.assertEqual(cli.reward_profile(args), profile)

    def test_checkpoint_inherits_stance_profile(self):
        metadata = {"reward_profile": "stance-v1", "reward_profile_schema": 1}
        for command, flag in (("demo", "--checkpoint"), ("evaluate", "--checkpoint"),
                              ("train", "--resume")):
            args = cli.build_parser().parse_args([command, flag, "policy.pt"])
            self.assertEqual(cli.reward_profile(args, metadata), "stance-v1")

    def test_reward_change_requires_training_and_preserves_observation_schema(self):
        old = {"reward_profile": "smooth-v1", "reward_profile_schema": 1}
        args = cli.build_parser().parse_args(
            ["train", "--resume", "policy.pt", "--reward-profile", "stance-v1"])
        self.assertEqual(cli.reward_profile(args, old), "stance-v1")
        self.assertEqual(cli.task_contract(args),
                         cli.task_contract(cli.build_parser().parse_args(["train"])))
        for command in ("demo", "evaluate"):
            args = cli.build_parser().parse_args(
                [command, "--checkpoint", "policy.pt", "--reward-profile", "stance-v1"])
            with self.assertRaisesRegex(ValueError, "must match"):
                cli.reward_profile(args, old)


if __name__ == "__main__":
    unittest.main()
