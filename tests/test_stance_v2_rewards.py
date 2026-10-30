"""Physical mirror conventions and upright shaping, without changing v1 policy inputs."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifting_lab import cli, mdp
from lifting_lab.env_cfg import make_env_cfg
from mjlab.envs import mdp as builtin
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from test_stance_rewards import fixture


def failure_term(env):
    return env.failure


def timeout_term(env):
    return env.timeout


class StanceV2NumericalTests(unittest.TestCase):
    def test_upright_known_tilt_heading_invariance_and_inversion(self):
        env = fixture(4)
        angle = math.radians(10)
        # Gravity direction relative to trunk: pitch and roll have identical
        # tilt; varying horizontal bearing leaves heading unpenalized.
        env.scene["robot"].data.projected_gravity_b = torch.tensor([
            [0., 0., -1.], [math.sin(angle), 0., -math.cos(angle)],
            [0., math.sin(angle), -math.cos(angle)], [0., 0., 1.]])
        result = mdp.trunk_tilt_cost(env)
        torch.testing.assert_close(result[:3], torch.tensor([0., 1., 1.]), atol=2e-6, rtol=2e-6)
        self.assertAlmostEqual(float(result[3]), 2 / (1 - math.cos(angle)), places=4)

    def test_mirrored_legs_use_physical_axis_signs_and_keep_bent_knees(self):
        env = fixture(2)
        positions = env.scene["robot"].data.joint_pos
        positions[1, 10:16] = torch.tensor([-.3, .2, .15, .7, -.35, -.1])
        positions[1, 16:22] = torch.tensor([-.3, -.2, -.15, .7, -.35, .1])
        torch.testing.assert_close(mdp.leg_symmetry_cost(env), torch.zeros(2))
        # Equal signs for roll are physically asymmetric, unlike pitch/knee.
        positions[1, 17] = .2
        self.assertGreater(float(mdp.leg_symmetry_cost(env)[1]), 0)

    def test_five_degree_deadband_and_ten_degree_excess_scale(self):
        env = fixture(3)
        positions = env.scene["robot"].data.joint_pos
        positions[0, 10:16] += math.radians(4)
        positions[1, 10:16] += math.radians(5)
        positions[2, 10:16] += math.radians(15)
        torch.testing.assert_close(mdp.leg_symmetry_cost(env), torch.tensor([0., 0., 1.]),
                                   atol=2e-6, rtol=2e-6)

    def test_arm_and_head_motion_does_not_change_leg_cost(self):
        env = fixture(2)
        env.scene["robot"].data.joint_pos[1, :10] += 1.5
        torch.testing.assert_close(mdp.leg_symmetry_cost(env), torch.zeros(2))

    def test_leg_names_must_be_complete_and_in_requested_order(self):
        for wrong in (lambda names, **kwargs: (list(range(5)), names[:5]),
                      lambda names, **kwargs: (list(range(6)), tuple(reversed(names)))):
            env = fixture()
            env.scene["robot"].find_joints = wrong
            with self.assertRaises(ValueError):
                mdp.leg_symmetry_cost(env)

    def test_parameter_validation_and_batch_finiteness(self):
        env = fixture(8)
        env.scene["robot"].data.projected_gravity_b = torch.tensor([0., 0., -1.]).repeat(8, 1)
        for function in (mdp.trunk_tilt_cost, mdp.leg_symmetry_cost):
            result = function(env)
            self.assertEqual(result.shape, (8,))
            self.assertTrue(bool(torch.isfinite(result).all()))
            for value in (0., -1., float("inf"), float("nan")):
                with self.subTest(function=function.__name__, value=value), self.assertRaises(ValueError):
                    function(env, scale_deg=value)
        for value in (-1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                mdp.leg_symmetry_cost(env, deadband_deg=value)

    def test_failure_reward_excludes_timeout_using_real_managers(self):
        env = SimpleNamespace(num_envs=4, device="cpu",
            failure=torch.tensor([False, False, True, True]),
            timeout=torch.tensor([False, True, False, True]))
        env.termination_manager = TerminationManager({
            "fail": TerminationTermCfg(func=failure_term),
            "timeout": TerminationTermCfg(func=timeout_term, time_out=True)}, env)
        env.termination_manager.compute()
        torch.testing.assert_close(builtin.is_terminated(env), torch.tensor([0., 0., 1., 1.]))
        manager = RewardManager({"failure": RewardTermCfg(func=builtin.is_terminated, weight=-100.)}, env)
        torch.testing.assert_close(manager.compute(.02), torch.tensor([0., 0., -2., -2.]))


class StanceV2CompatibilityTests(unittest.TestCase):
    def test_v2_adds_only_three_terms_and_keeps_all_v1_terms(self):
        v1 = make_env_cfg(support_mode="forearms", reward_profile="stance-v1")
        before = deepcopy(v1.rewards)
        v2 = make_env_cfg(support_mode="forearms", reward_profile="stance-v2")
        self.assertEqual(set(v2.rewards) - set(v1.rewards), {"trunk_tilt", "leg_symmetry", "failure"})
        for name, term in v1.rewards.items():
            self.assertEqual(v2.rewards[name], term)
        self.assertEqual(v1.rewards, before)
        for key in ("actions", "observations", "terminations", "events", "sim",
                    "decimation", "support_grace_steps", "episode_length_s"):
            self.assertEqual(getattr(v2, key), getattr(v1, key), key)
        self.assertEqual(v2.scene.sensors, v1.scene.sensors)
        self.assertIs(v2.rewards["failure"].func, builtin.is_terminated)
        self.assertLess(v2.rewards["trunk_tilt"].weight, 0)
        self.assertLess(v2.rewards["leg_symmetry"].weight, 0)

    def test_saved_profiles_remain_loadable_and_only_training_can_switch(self):
        for profile in ("baseline", "smooth-v1", "stance-v1", "stance-v2"):
            metadata = {"reward_profile": profile, "reward_profile_schema": 1}
            for command, flag in (("demo", "--checkpoint"), ("evaluate", "--checkpoint"),
                                  ("train", "--resume")):
                args = cli.build_parser().parse_args([command, flag, "policy.pt"])
                self.assertEqual(cli.reward_profile(args, metadata), profile)
        args = cli.build_parser().parse_args(
            ["train", "--resume", "policy.pt", "--reward-profile", "stance-v2"])
        old = {"reward_profile": "stance-v1", "reward_profile_schema": 1}
        self.assertEqual(cli.reward_profile(args, old), "stance-v2")
        args = cli.build_parser().parse_args(
            ["demo", "--checkpoint", "policy.pt", "--reward-profile", "stance-v2"])
        with self.assertRaisesRegex(ValueError, "must match"):
            cli.reward_profile(args, old)

    def test_pregrasp_rejects_v2_and_default_stays_baseline(self):
        with self.assertRaises(ValueError):
            make_env_cfg(support_mode="pregrasp", reward_profile="stance-v2")
        for command in ("smoke", "demo", "train", "evaluate"):
            self.assertEqual(cli.reward_profile(cli.build_parser().parse_args([command])), "baseline")


if __name__ == "__main__":
    unittest.main()
