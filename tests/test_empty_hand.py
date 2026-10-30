"""Empty-hand contracts and physical directions; never allocates a CUDA world."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifting_lab import empty_hand as task
from lifting_lab import empty_hand_cli as entry
from lifting_lab import mdp
from lifting_lab.env_cfg import make_env_cfg
from lifting_lab.model import forearm_keyframe, robot_mass_kg
from mjlab.envs import mdp as builtin
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg


class Scene(dict):
    pass


def fixture(batch=1, reverse_contacts=False):
    names = ["Head_Yaw", "Head_Pitch"]
    for side in ("Left", "Right"):
        names += [f"{side}_{joint}" for joint in
                  ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")]
    for side in ("Left", "Right"):
        names += [f"{side}_{joint}" for joint in
                  ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]
    keyframe = forearm_keyframe()
    nominal = torch.tensor([keyframe.joint_pos.get(name, 0.) for name in names]).repeat(batch, 1)
    root = torch.zeros(batch, 13)
    root[:, 2] = keyframe.pos[2]
    root[:, 3] = 1
    sole_names = ("left_foot_sole", "right_foot_sole")

    def find_joints(patterns, preserve_order=True):
        assert preserve_order
        ids = [i for pattern in patterns for i, name in enumerate(names) if re.fullmatch(pattern, name)]
        return ids, [names[i] for i in ids]

    def find_sites(patterns, preserve_order=True):
        assert preserve_order
        ids = [sole_names.index(name) for name in patterns]
        return ids, [sole_names[i] for i in ids]

    data = SimpleNamespace(
        joint_pos=nominal.clone(), default_joint_pos=nominal, joint_vel=torch.zeros(batch, 22),
        qfrc_actuator=torch.zeros(batch, 22), default_root_state=root,
        root_link_pos_w=root[:, :3].clone(), root_link_ang_vel_b=torch.zeros(batch, 3),
        root_link_lin_vel_b=torch.zeros(batch, 3),
        projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(batch, 1),
        site_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(batch, 2, 1),
        site_lin_vel_w=torch.zeros(batch, 2, 3))
    primary = ["robot/left_foot_link", "robot/right_foot_link"]
    if reverse_contacts:
        primary.reverse()
    contact = SimpleNamespace(primary_names=primary, data=SimpleNamespace(
        found=torch.ones(batch, 2), force=torch.tensor([0., 0., -100.]).repeat(batch, 2, 1)))
    scene = Scene(robot=SimpleNamespace(data=data, joint_names=names,
                                       find_joints=find_joints, find_sites=find_sites),
                  feet_ground_contact=contact)
    scene.env_origins = torch.zeros(batch, 3)
    return SimpleNamespace(num_envs=batch, device="cpu", scene=scene,
                           action_manager=SimpleNamespace(action=torch.zeros(batch, 22)))


class EmptyHandObservationTests(unittest.TestCase):
    def test_actor_is_exactly_72_proprioceptive_features_without_a_payload(self):
        env = fixture(2)
        r = env.scene["robot"].data
        r.root_link_ang_vel_b[:] = torch.tensor([.1, .2, .3])
        r.joint_pos += .01
        r.joint_vel[:] = .02
        env.action_manager.action[:] = .03
        actor = task.observation(env)
        expected = torch.cat((r.root_link_ang_vel_b, r.projected_gravity_b,
                              r.joint_pos-r.default_joint_pos, r.joint_vel,
                              env.action_manager.action), dim=-1)
        self.assertNotIn("payload", env.scene)
        self.assertEqual(tuple(actor.shape), (2, 72))
        torch.testing.assert_close(actor, expected)
        self.assertTrue(bool(torch.isfinite(actor).all()))

    def test_velocity_and_height_are_critic_only_and_height_uses_env_origin(self):
        env = fixture(2)
        actor_before = task.observation(env).clone()
        r = env.scene["robot"].data
        r.root_link_lin_vel_b[:] = torch.tensor([.2, -.1, .05])
        env.scene.env_origins[1, 2] = 3.
        r.root_link_pos_w[1, 2] += 3.
        critic = task.observation(env, privileged=True)
        self.assertEqual(tuple(critic.shape), (2, 76))
        torch.testing.assert_close(task.observation(env), actor_before)
        torch.testing.assert_close(critic[:, :72], actor_before)
        torch.testing.assert_close(critic[:, 72:75], r.root_link_lin_vel_b)
        torch.testing.assert_close(critic[:, 75], torch.full((2,), forearm_keyframe().pos[2]))


class EmptyHandRewardDirectionTests(unittest.TestCase):
    def test_foot_order_and_reaction_sign_use_names(self):
        env = fixture(reverse_contacts=True)
        env.scene["feet_ground_contact"].data.force[:] = torch.tensor(
            [[[3., -4., -60.], [-2., 5., -140.]]])
        env.scene["feet_ground_contact"].data.found[:] = torch.tensor([[0., 1.]])
        normal, reaction, contact = task.feet(env)
        torch.testing.assert_close(normal, torch.ones(1, 2))
        torch.testing.assert_close(reaction, torch.tensor([[[2., -5., 140.], [-3., 4., 60.]]]))
        torch.testing.assert_close(contact, torch.tensor([[True, False]]))

    def test_unknown_or_duplicate_foot_names_are_rejected(self):
        for names in (("left_foot_link", "left_foot_link"), ("left_foot_link", "other")):
            env = fixture()
            env.scene["feet_ground_contact"].primary_names = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                task.feet(env)

    def test_level_feet_score_better_than_tilt_or_inversion(self):
        env = fixture(3)
        angle = math.radians(30)
        env.scene["robot"].data.site_quat_w[1, 0] = torch.tensor(
            [math.cos(angle/2), math.sin(angle/2), 0., 0.])
        env.scene["robot"].data.site_quat_w[2, 0] = torch.tensor([0., 1., 0., 0.])
        cost = task.feet_flat(env)
        torch.testing.assert_close(cost, torch.tensor([0., 1-math.cos(angle), 2.]))
        self.assertLess(float(-20 * cost[2]), float(-20 * cost[1]))

    def test_load_share_penalizes_one_sided_or_absent_support(self):
        env = fixture(4)
        sensor = env.scene["feet_ground_contact"].data
        sensor.force[..., 2] = -torch.tensor([[100., 100.], [60., 140.], [5., 195.], [0., 0.]])
        sensor.found[3] = 0
        cost = task.load_share(env)
        torch.testing.assert_close(cost[:2], torch.zeros(2))
        self.assertGreater(float(cost[2]), 0.)
        self.assertGreater(float(cost[3]), 0.)
        torch.testing.assert_close(task.missing_feet(env), torch.tensor([0., 0., 0., 2.]))

    def test_contact_gated_slip_and_motion_costs(self):
        env = fixture(2)
        r = env.scene["robot"].data
        r.site_lin_vel_w[:, 0, 0] = 2.
        env.scene["feet_ground_contact"].data.found[1, 0] = 0
        torch.testing.assert_close(task.foot_slip(env), torch.tensor([4., 0.]))
        self.assertEqual(float(task.base_motion(env)[0]), 0.)
        r.root_link_ang_vel_b[1, 1] = .5
        r.root_link_lin_vel_b[1, 0] = .2
        self.assertGreater(float(task.base_motion(env)[1]), 0.)

    def test_lowering_arm_is_penalized_and_nominal_height_is_preferred(self):
        env = fixture(2)
        r = env.scene["robot"].data
        r.joint_pos[1, 3] += .5  # Named fixture index: Left_Shoulder_Roll.
        r.root_link_pos_w[1, 2] -= .05
        self.assertEqual(float(task.posture(env)[0]), 0.)
        self.assertGreater(float(task.posture(env)[1]), 0.)
        self.assertGreater(float(task.height_reward(env)[0]), float(task.height_reward(env)[1]))

    def test_robot_only_fall_checks_height_and_tilt_without_payload(self):
        env = fixture(3)
        r = env.scene["robot"].data
        r.root_link_pos_w[1, 2] = .37
        r.projected_gravity_b[2] = torch.tensor([math.sin(math.radians(50)), 0., -math.cos(math.radians(50))])
        torch.testing.assert_close(task.fell(env), torch.tensor([False, True, True]))


def failure(env):
    return env.failure


def timeout(env):
    return env.timeout


class EmptyHandConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.scene import Scene as NativeScene
        cls.cfg = task.make_empty_hand_cfg(num_envs=1)
        cls.scene = NativeScene(cls.cfg.scene, device="cpu")
        cls.model = cls.scene.compile()

    def test_compiled_model_has_one_robot_and_no_payload_body_geom_or_equality(self):
        self.assertEqual(set(self.cfg.scene.entities), {"robot"})
        self.assertIsNone(self.cfg.scene.spec_fn)
        self.assertEqual(self.model.neq, 0)
        self.assertEqual(sum(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 1)
        self.assertEqual(self.model.nu, 22)
        for category, count in (("body", self.model.nbody), ("geom", self.model.ngeom), ("joint", self.model.njnt)):
            self.assertFalse(any("payload" in getattr(self.model, category)(i).name.lower()
                                 for i in range(count)))
        self.assertAlmostEqual(float(self.model.body_mass.sum()), robot_mass_kg(), places=8)
        self.assertEqual([s.name for s in self.cfg.scene.sensors], ["feet_ground_contact"])

    def test_task_keeps_forearm_pose_action_scales_and_control_rate(self):
        loaded = make_env_cfg(num_envs=1, support_mode="forearms", reward_profile="stance-v2")
        self.assertEqual(self.cfg.scene.entities["robot"].init_state, loaded.scene.entities["robot"].init_state)
        self.assertEqual(self.cfg.actions, loaded.actions)
        self.assertEqual(self.cfg.sim, loaded.sim)
        self.assertAlmostEqual(self.cfg.sim.mujoco.timestep * self.cfg.decimation, .02)
        self.assertEqual(set(self.cfg.events), {"reset"})
        self.assertEqual(set(self.cfg.terminations), {"time_out", "fell", "nonfinite"})

    def test_old_loaded_task_configuration_is_not_mutated(self):
        before = make_env_cfg(num_envs=1, support_mode="forearms", reward_profile="stance-v2")
        task.make_empty_hand_cfg(num_envs=1)
        after = make_env_cfg(num_envs=1, support_mode="forearms", reward_profile="stance-v2")
        self.assertEqual(before.rewards, after.rewards)
        self.assertEqual(before.events, after.events)
        self.assertIn("payload", after.scene.entities)
        self.assertEqual(after.observations["actor"].terms["state"].func, mdp.task_observation)

    def test_failure_penalty_excludes_timeouts_in_real_managers(self):
        env = SimpleNamespace(num_envs=4, device="cpu", failure=torch.tensor([0, 0, 1, 1], dtype=torch.bool),
                              timeout=torch.tensor([0, 1, 0, 1], dtype=torch.bool))
        env.termination_manager = TerminationManager({
            "fell": TerminationTermCfg(func=failure),
            "timeout": TerminationTermCfg(func=timeout, time_out=True)}, env)
        env.termination_manager.compute()
        term = deepcopy(self.cfg.rewards["failure"])
        self.assertIs(term.func, builtin.is_terminated)
        manager = RewardManager({"failure": term}, env)
        torch.testing.assert_close(manager.compute(.02), torch.tensor([0., 0., -2., -2.]))


class EmptyHandCheckpointContractTests(unittest.TestCase):
    def checkpoint(self, directory, metadata):
        path = Path(directory) / "policy.pt"
        path.write_bytes(b"metadata-only test; never load these bytes as a model")
        path.with_name("experiment.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def test_matching_new_contract_accepts_and_has_no_payload(self):
        contract = entry.metadata_contract()
        self.assertEqual(contract["actor_observation_dim"], 72)
        self.assertEqual(contract["critic_observation_dim"], 76)
        self.assertFalse(contract["payload_entity_present"])
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory, {**contract, "runner": {}})
            self.assertEqual(entry.read_metadata(path), {**contract, "runner": {}})

    def test_every_contract_mismatch_is_rejected(self):
        wrong_values = {
            "task_schema": "forearms-contact-v1", "actor_observation_schema": "k1-load-state-97-v1",
            "critic_observation_schema": "k1-load-state-97-v1", "actor_observation_dim": 97,
            "critic_observation_dim": 97, "action_dim": 12, "control_dt_s": .01,
            "payload_entity_present": True, "reward_profile": "stance-v2",
            "action_contract": "torque control",
        }
        for key, value in wrong_values.items():
            with self.subTest(key=key), TemporaryDirectory() as directory:
                meta = {**entry.metadata_contract(), "runner": {}, key: value}
                path = self.checkpoint(directory, meta)
                with self.assertRaisesRegex(ValueError, key):
                    entry.read_metadata(path)

    def test_actual_hold_metadata_is_rejected_without_loading_weights(self):
        source = Path(__file__).resolve().parents[1] / "shared_model/stance-v2-20261029/experiment.json"
        metadata = json.loads(source.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory, metadata)
            with self.assertRaisesRegex(ValueError, "task_schema"):
                entry.read_metadata(path)

    def test_missing_runner_and_missing_checkpoint_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = self.checkpoint(directory, entry.metadata_contract())
            with self.assertRaisesRegex(ValueError, "runner"):
                entry.read_metadata(path)
            with self.assertRaises(FileNotFoundError):
                entry.read_metadata(Path(directory) / "absent.pt")


class FakeRunner:
    """Exercise the task orchestration without constructing a simulation."""
    def __init__(self):
        self.current_learning_iteration = 0
        self.loaded = False
        self.alg = SimpleNamespace(
            actor=SimpleNamespace(mlp=torch.nn.Sequential(torch.nn.Linear(72, 8),
                                                        torch.nn.ELU(), torch.nn.Linear(8, 22))),
            optimizer=SimpleNamespace(param_groups=[{"lr": 1.}]),
            update=lambda: {"loss": .1}, save=lambda: {"finite_parameter": torch.ones(1)})
        self.last_before_learn = None

    def learn(self, count, init_at_random_ep_len=False):
        assert init_at_random_ep_len is False
        final = list(self.alg.actor.mlp.modules())[-1]
        self.last_before_learn = (final.weight.detach().clone(), final.bias.detach().clone())
        start = self.current_learning_iteration
        for iteration in range(start, start + count):
            self.alg.update()
            self.current_learning_iteration = iteration

    def load(self, path, map_location):
        assert map_location == "cpu"
        self.loaded = True
        self.current_learning_iteration = json.loads(Path(path).read_text())["iter"]

    def save(self, path):
        Path(path).write_text(json.dumps({"iter": self.current_learning_iteration}), encoding="utf-8")


class EmptyHandTrainingOrchestrationTests(unittest.TestCase):
    def args(self, output, resume=None, iterations=3, num_envs=4):
        return SimpleNamespace(output=output, resume=resume, seed=11, iterations=iterations,
                               num_envs=num_envs, learning_rate=.0003, joint_jitter=.015,
                               reset_speed=.02, push_speed=0.)

    def run_mock_train(self, args):
        runner = FakeRunner()
        closed = []
        env = SimpleNamespace(device="cpu", close=lambda: closed.append(True))
        with patch.object(entry, "environment", return_value=env), \
             patch.object(entry, "make_runner", return_value=runner) as make, \
             patch.object(entry, "runtime_info", return_value={"source_sha256": "fixed-source"}):
            entry.train(args)
        self.assertEqual(closed, [True])
        return runner, make.call_args.args[1]

    def test_fresh_mean_is_zero_before_learning_and_budget_matches_completed_steps(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            runner, config = self.run_mock_train(self.args(output))
            for value in runner.last_before_learn:
                torch.testing.assert_close(value, torch.zeros_like(value))
            self.assertFalse(runner.loaded)
            self.assertEqual(config["actor"]["distribution_cfg"]["init_std"], .15)
            result = json.loads((output / "experiment.json").read_text())
            self.assertEqual(result["phase_start_iteration"], 0)
            self.assertEqual(result["last_completed_iteration"], 2)
            self.assertEqual(result["actual_phase_steps"], 3 * 24 * 4)
            self.assertEqual(result["actual_cumulative_steps"], 3 * 24 * 4)
            self.assertFalse(result["hardware_ready"])
            self.assertTrue((output / "completed.json").is_file())

    def test_resume_midphase_uses_saved_iteration_and_previous_batch_size(self):
        from lifting_lab.cli import _runner_config
        with TemporaryDirectory() as directory:
            previous = Path(directory) / "previous"
            previous.mkdir()
            checkpoint = previous / "model_4.pt"
            checkpoint.write_text(json.dumps({"iter": 4}))
            old = {**entry.metadata_contract(), "runner": _runner_config(11), "seed": 11,
                   "phase_start_iteration": 0, "prior_env_steps": 0, "num_envs": 8,
                   "planned_phase_steps": 10 * 24 * 8}
            checkpoint.with_name("experiment.json").write_text(json.dumps(old))
            output = Path(directory) / "resumed"
            output.mkdir()
            runner, _ = self.run_mock_train(self.args(output, checkpoint, iterations=2, num_envs=4))
            self.assertTrue(runner.loaded)
            result = json.loads((output / "experiment.json").read_text())
            self.assertEqual(result["phase_start_iteration"], 5)
            self.assertEqual(result["last_completed_iteration"], 6)
            self.assertEqual(result["prior_env_steps"], 5 * 24 * 8)
            self.assertEqual(result["actual_cumulative_steps"], 5 * 24 * 8 + 2 * 24 * 4)


if __name__ == "__main__":
    unittest.main()
