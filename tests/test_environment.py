"""Opt-in bounded simulator checks; GPU tests must be run without training jobs.

CPU: set LIFTING_TEST_ENV=1, then unittest discovery.
GPU parity: set LIFTING_TEST_GPU=1 using the CUDA runtime, then discovery.
The native model/metric tests run by default without either option.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab.env_cfg import make_env_cfg
from lifting_lab import mdp
from lifting_lab.model import robot_mass_kg


class EnvironmentConfigurationTests(unittest.TestCase):
    def test_invalid_environment_options(self):
        for kwargs in ({"num_envs": 0}, {"num_envs": True}, {"num_envs": 1.5},
                       {"episode_length_s": 0}, {"episode_length_s": np.inf},
                       {"reference_height_m": np.nan}, {"payload_mass_kg": -1},
                       {"reset_xy_range_m": -0.01}, {"reset_yaw_range_rad": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_env_cfg(**kwargs)

    def test_zero_mass_request_keeps_actual_floor_visible(self):
        cfg = make_env_cfg(payload_mass_kg=0.0, num_envs=1)
        self.assertEqual(cfg.requested_payload_mass_kg, 0.0)
        self.assertEqual(cfg.actual_payload_mass_kg, 0.001)

    def test_twenty_seconds_is_one_thousand_control_steps(self):
        cfg = make_env_cfg(num_envs=1)
        self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, 0.02)
        self.assertEqual(round(cfg.episode_length_s / (cfg.sim.mujoco.timestep * cfg.decimation)), 1000)

    def test_play_does_not_remove_failure_criteria_or_extend_horizon(self):
        train = make_env_cfg(num_envs=1)
        play = make_env_cfg(num_envs=1, play=True)
        self.assertEqual(train.episode_length_s, play.episode_length_s)
        self.assertEqual(set(train.terminations), set(play.terminations))
        self.assertIn("fell", play.terminations)
        self.assertIn("grip_failed", play.terminations)

    def test_observation_ablation_does_not_change_physics_configuration(self):
        observed = make_env_cfg(payload_mass_kg=0.025 * robot_mass_kg(), num_envs=1)
        blind = make_env_cfg(payload_mass_kg=0.025 * robot_mass_kg(), num_envs=1,
                             mass_observed=False)
        self.assertEqual(observed.actual_payload_mass_kg, blind.actual_payload_mass_kg)
        self.assertEqual(observed.reference_height_m, blind.reference_height_m)
        np.testing.assert_array_equal(observed.scene.entities["payload"].spec_fn().compile().body_mass,
                                      blind.scene.entities["payload"].spec_fn().compile().body_mass)


@unittest.skipUnless(os.environ.get("LIFTING_TEST_ENV") == "1",
                     "Set LIFTING_TEST_ENV=1 to run bounded CPU simulation checks")
class CpuEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv

        torch.set_num_threads(2)
        cfg = make_env_cfg(num_envs=2, payload_mass_kg=0.5)
        cfg.seed = 123
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "env"):
            cls.env.close()

    def setUp(self):
        self.obs, _ = self.env.reset(seed=123)

    def test_reset_and_step_shapes_and_finiteness(self):
        self.assertEqual(self.env.action_manager.total_action_dim, 22)
        self.assertEqual(tuple(self.obs["actor"].shape), (2, 97))
        self.assertEqual(tuple(self.obs["critic"].shape), (2, 97))
        for _ in range(4):
            obs, rewards, terminated, truncated, _ = self.env.step(torch.zeros((2, 22)))
            for values in (obs["actor"], obs["critic"], rewards, self.env.sim.data.qpos,
                           self.env.sim.data.qvel):
                self.assertTrue(torch.isfinite(values).all().item())
            self.assertEqual(tuple(rewards.shape), (2,))
            self.assertEqual(tuple(terminated.shape), (2,))
            self.assertEqual(tuple(truncated.shape), (2,))

    def test_seeded_reset_preserves_grasp_and_reproduces_state(self):
        first = self.env.sim.data.qpos.clone()
        first_obs = self.obs["actor"].clone()
        self.env.reset(seed=456)
        changed = self.env.sim.data.qpos.clone()
        self.assertGreater(torch.max(torch.abs(changed - first)).item(), 1e-5)
        obs, _ = self.env.reset(seed=123)
        torch.testing.assert_close(self.env.sim.data.qpos.clone(), first, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(obs["actor"], first_obs, rtol=1e-6, atol=2e-6)
        self.assertLess(torch.linalg.vector_norm(mdp.grip_displacement(self.env), dim=-1).max().item(), 2e-6)

    def test_mass_blind_masks_only_mass_feature(self):
        observed = mdp.task_observation(self.env, mass_observed=True)
        blind = mdp.task_observation(self.env, mass_observed=False)
        self.assertEqual(observed.shape, blind.shape)
        torch.testing.assert_close(observed[:, :-3], blind[:, :-3], rtol=0, atol=0)
        torch.testing.assert_close(observed[:, -2:], blind[:, -2:], rtol=0, atol=0)
        self.assertTrue(torch.all(blind[:, -3] == 0).item())
        expected = self.env.cfg.actual_payload_mass_kg / robot_mass_kg()
        torch.testing.assert_close(observed[:, -3], torch.full((2,), expected))

    def test_runtime_mass_and_measured_metrics_match_configuration(self):
        model = self.env.sim.mj_model
        payload_id = model.body("payload/box").id
        self.assertAlmostEqual(model.body_mass[payload_id], 0.5)
        torch.testing.assert_close(self.env.sim.model.body_mass[..., payload_id],
                                   torch.full_like(self.env.sim.model.body_mass[..., payload_id], 0.5))
        metrics = mdp.task_metrics(self.env)
        for key, values in metrics.items():
            self.assertEqual(tuple(values.shape), (2,), key)
            self.assertTrue(torch.isfinite(values).all().item(), key)
        self.assertLess(metrics["grip_error_m"].max().item(), 2e-6)
        self.assertLess(metrics["height_error_m"].abs().max().item(), 2e-6)


@unittest.skipUnless(os.environ.get("LIFTING_TEST_GPU") == "1",
                     "Set LIFTING_TEST_GPU=1; do not run alongside GPU training")
class CpuGpuParityTests(unittest.TestCase):
    def test_short_zero_action_rollout_is_physically_close_not_bitwise_identical(self):
        from mjlab.envs import ManagerBasedRlEnv

        self.assertTrue(torch.cuda.is_available(), "Explicit GPU test requested without CUDA")
        outputs = []
        for device in ("cpu", "cuda:0"):
            cfg = make_env_cfg(num_envs=1, payload_mass_kg=0.5, reset_xy_range_m=0.0,
                               reset_yaw_range_rad=0.0, reset_speed_range_m_s=0.0)
            cfg.seed = 123
            env = ManagerBasedRlEnv(cfg, device=device)
            try:
                env.reset(seed=123)
                for _ in range(3):
                    obs, rewards, terminated, _, _ = env.step(torch.zeros((1, 22), device=device))
                    self.assertTrue(torch.isfinite(obs["actor"]).all().item())
                    self.assertTrue(torch.isfinite(rewards).all().item())
                outputs.append((env.sim.data.qpos.detach().cpu().numpy().copy(),
                                terminated.detach().cpu().numpy().copy()))
            finally:
                env.close()
        # Contact solvers and reduction orders differ; long-horizon trajectory
        # equality is neither expected nor claimed by this short sanity check.
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0.02, atol=0.01)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


if __name__ == "__main__":
    unittest.main()
