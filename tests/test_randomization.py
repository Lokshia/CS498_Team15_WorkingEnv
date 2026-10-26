"""Per-environment payload mass/COM, payload pose jitter and push events."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.managers.event_manager import RecomputeLevel
from lifting_lab import cli, mdp
from lifting_lab.env_cfg import make_env_cfg
from lifting_lab.model import MIN_PAYLOAD_MASS_KG, PayloadCfg, robot_mass_kg


class RandomizationConfigTests(unittest.TestCase):
    def test_defaults_add_no_events_and_keep_the_fixed_mass_task(self):
        for mode, events in (("forearms", {"reset", "support_history"}), ("pregrasp", {"reset"})):
            with self.subTest(mode=mode):
                cfg = make_env_cfg(num_envs=1, support_mode=mode)
                self.assertEqual(set(cfg.events), events)
                self.assertIsNone(cfg.payload_mass_range_kg)
                self.assertIsNone(cfg.push_interval_s)
                self.assertEqual(cfg.payload_com_offset_m, (0.0, 0.0, 0.0))
                self.assertEqual(cfg.payload_xy_jitter_m, 0.0)
                self.assertEqual(cfg.events["reset"].params["payload_xy_jitter"], 0.0)
                self.assertEqual(cfg.payload_half_size_m, PayloadCfg().half_size)

    def test_randomization_events_are_registered_with_their_modes(self):
        cfg = make_env_cfg(num_envs=1, support_mode="forearms", payload_mass_kg=0.5,
                           payload_mass_range_kg=(0.0, 1.0), payload_com_offset_m=(0.02, 0.03, 0.0),
                           payload_xy_jitter_m=0.01, payload_yaw_jitter_rad=0.02,
                           push_interval_s=(2.0, 4.0), push_velocity_m_s=0.3)
        self.assertEqual(cfg.actual_payload_mass_kg, 0.5)
        self.assertEqual(cfg.payload_mass_range_kg, (0.0, 1.0))
        inertial = cfg.events["payload_inertial"]
        self.assertEqual(inertial.mode, "reset")
        self.assertIs(inertial.func, mdp.randomize_payload_inertial)
        self.assertEqual(inertial.params, {"mass_range_kg": (0.0, 1.0), "com_offset_m": (0.02, 0.03, 0.0)})
        push = cfg.events["push"]
        self.assertEqual(push.mode, "interval")
        self.assertEqual(push.interval_range_s, (2.0, 4.0))
        self.assertFalse(push.is_global_time)
        self.assertEqual(push.params["velocity_range"], {"x": (-0.3, 0.3), "y": (-0.3, 0.3)})
        self.assertEqual(push.params["asset_cfg"].name, "robot")
        self.assertEqual(cfg.events["reset"].params["payload_xy_jitter"], 0.01)
        self.assertEqual(cfg.events["reset"].params["payload_yaw_jitter"], 0.02)
        com_only = make_env_cfg(num_envs=1, support_mode="forearms", payload_com_offset_m=(0.0, 0.01, 0.0))
        self.assertIsNone(com_only.events["payload_inertial"].params["mass_range_kg"])

    def test_invalid_randomization_options(self):
        for kwargs in ({"payload_mass_range_kg": (1.0, 0.5)}, {"payload_mass_range_kg": (-0.1, 1.0)},
                       {"payload_mass_range_kg": (0.0, float("inf"))},
                       {"payload_com_offset_m": (0.10, 0.0, 0.0)}, {"payload_com_offset_m": (0.0, -0.01, 0.0)},
                       {"payload_com_offset_m": (0.01, 0.01)},
                       {"payload_xy_jitter_m": 0.02}, {"payload_yaw_jitter_rad": 0.05},
                       {"payload_xy_jitter_m": -0.01},
                       {"push_interval_s": (2.0, 4.0)}, {"push_velocity_m_s": 0.3},
                       {"push_interval_s": (0.0, 1.0), "push_velocity_m_s": 0.3},
                       {"push_interval_s": (3.0, 2.0), "push_velocity_m_s": 0.3},
                       {"push_interval_s": (1.0, 2.0), "push_velocity_m_s": -0.3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_env_cfg(num_envs=1, support_mode="forearms", **kwargs)


def inertial_env(num_envs=3, nominal=0.5, half_size=(0.10, 0.20, 0.06), nbody=4, box=2):
    model = SimpleNamespace(body_mass=torch.full((num_envs, nbody), 7.0),
                            body_ipos=torch.full((num_envs, nbody, 3), 9.0),
                            body_inertia=torch.full((num_envs, nbody, 3), 11.0))
    return SimpleNamespace(
        num_envs=num_envs, device="cpu", sim=SimpleNamespace(model=model),
        cfg=SimpleNamespace(actual_payload_mass_kg=nominal, payload_half_size_m=half_size),
        scene={"payload": SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([box])))})


class PayloadInertialTests(unittest.TestCase):
    def test_each_environment_gets_its_own_mass_centre_and_box_inertia(self):
        torch.manual_seed(0)
        env = inertial_env()
        mdp.randomize_payload_inertial(env, None, mass_range_kg=(0.2, 1.2), com_offset_m=(0.03, 0.05, 0.0))
        mass = env.sim.model.body_mass[:, 2]
        self.assertTrue(((mass >= 0.2) & (mass <= 1.2)).all())
        self.assertGreater(mass.max() - mass.min(), 0.0)
        ipos = env.sim.model.body_ipos[:, 2]
        self.assertTrue((ipos[:, 0].abs() <= 0.03).all() and (ipos[:, 1].abs() <= 0.05).all())
        self.assertTrue((ipos[:, 2] == 0).all())
        expected = PayloadCfg(mass_kg=1.0).diagonal_inertia
        torch.testing.assert_close(env.sim.model.body_inertia[:, 2],
                                   mass[:, None] * torch.tensor(expected), rtol=1e-5, atol=1e-8)
        # Other bodies and non-selected rows are untouched.
        self.assertTrue((env.sim.model.body_mass[:, [0, 1, 3]] == 7.0).all())
        self.assertTrue((env.sim.model.body_ipos[:, [0, 1, 3]] == 9.0).all())

    def test_selected_rows_only_nominal_mass_without_range_and_one_gram_floor(self):
        env = inertial_env()
        mdp.randomize_payload_inertial(env, torch.tensor([1]), mass_range_kg=None, com_offset_m=(0.0, 0.0, 0.0))
        self.assertEqual(env.sim.model.body_mass[:, 2].tolist(), [7.0, 0.5, 7.0])
        self.assertEqual(env.sim.model.body_ipos[1, 2].tolist(), [0.0, 0.0, 0.0])
        mdp.randomize_payload_inertial(env, None, mass_range_kg=(0.0, 0.0), com_offset_m=(0.0, 0.0, 0.0))
        self.assertTrue((env.sim.model.body_mass[:, 2] == MIN_PAYLOAD_MASS_KG).all())
        self.assertGreater(env.sim.model.body_inertia[:, 2].min(), 0.0)

    def test_declares_per_world_fields_and_full_constant_recompute(self):
        fields = set(mdp.randomize_payload_inertial.model_fields)
        self.assertTrue({"body_mass", "body_ipos", "body_inertia", "body_subtreemass",
                         "body_invweight0", "dof_invweight0"} <= fields)
        self.assertEqual(mdp.randomize_payload_inertial.recompute, RecomputeLevel.set_const)

    def test_observed_mass_reads_shared_or_per_world_model_rows(self):
        shared = inertial_env(num_envs=3)
        shared.sim.model.body_mass = torch.tensor([[0.0, 1.0, 0.25, 2.0]])
        self.assertEqual(mdp.payload_mass_kg(shared).tolist(), [0.25, 0.25, 0.25])
        per_world = inertial_env(num_envs=2)
        per_world.sim.model.body_mass = torch.tensor([[0.0, 1.0, 0.25, 2.0], [0.0, 1.0, 0.75, 2.0]])
        self.assertEqual(mdp.payload_mass_kg(per_world).tolist(), [0.25, 0.75])


class CliRandomizationTests(unittest.TestCase):
    def test_flags_parse_for_training_commands_but_not_evaluate(self):
        args = cli.build_parser().parse_args([
            "train", "--mass-range-fraction", "0", "0.05", "--com-offset-m", "0.02", "0.03", "0",
            "--payload-xy-jitter-m", "0.01", "--payload-yaw-jitter-rad", "0.02",
            "--push-interval-s", "3", "5", "--push-velocity-m-s", "0.3"])
        self.assertEqual(args.mass_range_fraction, [0.0, 0.05])
        self.assertEqual(cli._randomization(args), {
            "mass_range_fraction": [0.0, 0.05], "com_offset_m": [0.02, 0.03, 0.0],
            "payload_xy_jitter_m": 0.01, "payload_yaw_jitter_rad": 0.02,
            "push_interval_s": [3.0, 5.0], "push_velocity_m_s": 0.3})
        plain = cli.build_parser().parse_args(["smoke"])
        self.assertEqual(cli._randomization(plain), {
            "mass_range_fraction": None, "com_offset_m": [0.0, 0.0, 0.0], "payload_xy_jitter_m": 0.0,
            "payload_yaw_jitter_rad": 0.0, "push_interval_s": None, "push_velocity_m_s": 0.0})
        evaluate = cli.build_parser().parse_args(["evaluate"])
        self.assertFalse(hasattr(evaluate, "push_velocity_m_s"))
        self.assertEqual(cli._randomization(evaluate)["push_interval_s"], None)
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["train", "--mass", "0.5", "--mass-range-fraction", "0", "0.05"])

    def test_push_flags_must_be_paired_and_curriculum_excludes_mass_range(self):
        for argv in (["smoke", "--push-interval-s", "3", "5"], ["smoke", "--push-velocity-m-s", "0.3"],
                     ["train", "--curriculum", "--mass-range-fraction", "0", "0.05"]):
            with self.subTest(argv=argv):
                with patch.object(sys, "argv", ["lifting_lab", *argv]):
                    with patch.object(cli, "smoke"), patch.object(cli, "train"):
                        with self.assertRaises(SystemExit):
                            cli.main()
        args = cli.build_parser().parse_args(["smoke", "--push-interval-s", "3", "5"])
        with self.assertRaisesRegex(ValueError, "push-velocity"):
            cli._randomization(args)
        args = cli.build_parser().parse_args(["smoke", "--push-interval-s", "3", "5", "--push-velocity-m-s", "0"])
        with self.assertRaisesRegex(ValueError, "positive"):
            cli._randomization(args)

    def test_mass_range_uses_the_midpoint_as_nominal_mass(self):
        args = cli.build_parser().parse_args(["train", "--mass-range-fraction", "0", "0.05"])
        mass = cli._mass(args)
        robot = robot_mass_kg()
        self.assertAlmostEqual(mass["requested_payload_mass_kg"], 0.025 * robot)
        self.assertEqual(mass["mass_range_fraction"], [0.0, 0.05])
        self.assertAlmostEqual(mass["mass_range_kg"][1], 0.05 * robot)
        self.assertTrue(mass["nominal_mass_is_range_midpoint"])
        fixed = cli._mass(cli.build_parser().parse_args(["train", "--mass", "0.5"]))
        self.assertIsNone(fixed["mass_range_kg"])
        self.assertFalse(fixed["nominal_mass_is_range_midpoint"])

    def test_resume_inherits_checkpoint_randomization_unless_overridden(self):
        metadata = {"randomization": {"com_offset_m": [0.02, 0.02, 0.0], "payload_xy_jitter_m": 0.01,
                                      "payload_yaw_jitter_rad": 0.0, "push_interval_s": [3.0, 5.0],
                                      "push_velocity_m_s": 0.3, "mass_range_fraction": [0.0, 0.05]}}
        args = cli.build_parser().parse_args(["train", "--push-velocity-m-s", "0.1", "--push-interval-s", "1", "2"])
        cli._inherit_randomization(args, metadata)
        self.assertEqual(args.com_offset_m, [0.02, 0.02, 0.0])
        self.assertEqual(args.payload_xy_jitter_m, 0.01)
        self.assertEqual(args.push_velocity_m_s, 0.1)
        self.assertEqual(args.push_interval_s, [1.0, 2.0])
        self.assertIsNone(args.mass_range_fraction)
        bare = cli.build_parser().parse_args(["train"])
        cli._inherit_randomization(bare, {"runner": {}})
        self.assertIsNone(bare.com_offset_m)
        no_push = cli.build_parser().parse_args(["train"])
        cli._inherit_randomization(no_push, {"randomization": {
            "mass_range_fraction": None, "com_offset_m": [0.0, 0.0, 0.0], "payload_xy_jitter_m": 0.0,
            "payload_yaw_jitter_rad": 0.0, "push_interval_s": None, "push_velocity_m_s": 0.0}})
        self.assertIsNone(no_push.push_interval_s)
        self.assertIsNone(no_push.push_velocity_m_s)
        self.assertEqual(cli._randomization(no_push)["push_velocity_m_s"], 0.0)

    def test_push_protocol_shares_the_v1_contract_and_adds_a_disturbance_block(self):
        root = Path(cli.__file__).resolve().parents[2] / "configs"
        base = json.loads((root / "team_protocol.json").read_text(encoding="utf-8"))
        push = json.loads((root / "team_protocol_push.json").read_text(encoding="utf-8"))
        for key in ("support_mode", "task_schema", "observation_schema", "training_mass_fractions",
                    "evaluation_mass_fractions", "training_seeds", "evaluation_seeds",
                    "episodes_per_seed_condition", "episode_seconds", "success"):
            self.assertEqual(base[key], push[key], key)
        self.assertNotEqual(base["protocol_id"], push["protocol_id"])
        self.assertEqual(cli._protocol_conditions(base)["push_interval_s"], None)
        conditions = cli._protocol_conditions(push)
        self.assertEqual(conditions["push_interval_s"], [3.0, 5.0])
        self.assertEqual(conditions["push_velocity_m_s"], 0.3)
        args = cli.build_parser().parse_args(["evaluate", "--protocol", str(root / "team_protocol_push.json")])
        _, loaded = cli._evaluation_protocol(args)
        self.assertEqual(loaded["protocol_id"], "k1-forearms-contact-v1-push")

    def test_protocol_rejects_unknown_or_half_specified_conditions(self):
        args = cli.build_parser().parse_args(["evaluate"])
        _, protocol = cli._evaluation_protocol(args)
        with TemporaryDirectory() as directory:
            args.protocol = Path(directory) / "protocol.json"
            for block in ({"disturbance": {"push_interval_s": [3, 5]}},
                          {"disturbance": {"wind": 1}},
                          {"randomization": {"payload_mass_range_kg": [0, 1]}}):
                with self.subTest(block=block):
                    args.protocol.write_text(json.dumps({**protocol, **block}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        cli._evaluation_protocol(args)


@unittest.skipUnless(os.environ.get("LIFTING_TEST_ENV") == "1", "Enable bounded CPU world checks")
class RandomizedWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        torch.set_num_threads(2)
        cfg = make_env_cfg(num_envs=4, support_mode="forearms", payload_mass_kg=0.5,
                           payload_mass_range_kg=(0.1, 1.5), payload_com_offset_m=(0.02, 0.03, 0.0),
                           payload_xy_jitter_m=0.01, payload_yaw_jitter_rad=0.02,
                           push_interval_s=(0.1, 0.1), push_velocity_m_s=0.3)
        cfg.auto_reset = False
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_writes_per_world_payload_fields_that_the_observation_reports(self):
        env = self.env
        obs, _ = env.reset(seed=5)
        box = mdp._payload_body_id(env)
        model = env.sim.model
        self.assertEqual(tuple(model.body_mass.shape)[0], 4)
        mass = model.body_mass[:, box]
        self.assertGreater(mass.max() - mass.min(), 0.05)
        self.assertTrue(((mass >= 0.1) & (mass <= 1.5)).all())
        torch.testing.assert_close(model.body_subtreemass[:, box], mass)
        torch.testing.assert_close(obs["actor"][:, 94], mass / robot_mass_kg())
        self.assertTrue((model.body_ipos[:, box, 1].abs() <= 0.03).all())
        self.assertTrue(torch.isfinite(model.body_invweight0).all())
        before = mass.clone()
        env.reset(env_ids=torch.tensor([2]))
        after = model.body_mass[:, box]
        self.assertEqual((before != after).tolist(), [False, False, True, False])

    def test_jittered_starts_are_free_of_forbidden_contact_and_reach_support(self):
        env = self.env
        env.reset(seed=6)
        seen = torch.zeros(4, dtype=torch.bool)
        for step in range(3):
            _, _, terminated, _, _ = env.step(torch.zeros(4, 22))
            values = mdp.task_metrics(env)
            self.assertFalse(values["payload_disallowed_contact"].any(), step)
            self.assertFalse(terminated.any(), step)
            seen |= values["payload_forearm_supported"]
        self.assertTrue(seen.all())

    def test_interval_push_kicks_the_robot_root_on_the_fifth_step(self):
        env = self.env
        env.reset(seed=7)
        robot = env.scene["robot"].data
        velocities = []
        for _ in range(5):
            env.step(torch.zeros(4, 22))
            velocities.append(robot.root_link_lin_vel_w[:, :2].clone())
        # Natural drift changes little from step to step; the 0.1 s interval
        # fires inside the fifth step and adds up to 0.3 m/s per axis.
        natural = velocities[3] - velocities[2]
        pushed = velocities[4] - velocities[3]
        self.assertLess(natural.abs().max(), 0.05)
        self.assertGreater((pushed - natural).abs().max(), 0.05)
        self.assertTrue(torch.isfinite(env.sim.data.qvel).all())


if __name__ == "__main__":
    unittest.main()
