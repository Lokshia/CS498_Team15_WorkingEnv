"""Native contact checks for the unconstrained forearm-support preview."""
from pathlib import Path
import os
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifting_lab.model import (
    PayloadCfg, forearm_keyframe, initial_forearm_geometry,
    make_robot_cfg, make_payload_cfg,
)
from lifting_lab.env_cfg import make_env_cfg


class ForearmModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.scene import Scene, SceneCfg
        cls.scene = Scene(SceneCfg(num_envs=1, entities={
            "robot": make_robot_cfg("forearms"),
            "payload": make_payload_cfg(PayloadCfg(mass_kg=.5), "forearms"),
        }), device="cpu")
        cls.model = cls.scene.compile()

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        state = forearm_keyframe()
        center, _ = initial_forearm_geometry()
        for jid in range(self.model.njnt):
            name = self.model.joint(jid).name
            adr = self.model.jnt_qposadr[jid]
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                self.data.qpos[adr:adr+7] = ((*state.pos, *state.rot)
                    if name.startswith("robot/") else (*center, 1, 0, 0, 0))
                if name.startswith("payload/"):
                    self.payload_adr = adr
                    self.payload_dof = self.model.jnt_dofadr[jid]
            else:
                value = state.joint_pos.get(name.split("/", 1)[1], 0.0)
                self.data.qpos[adr] = value
                self.assertGreaterEqual(value, self.model.jnt_range[jid, 0])
                self.assertLessEqual(value, self.model.jnt_range[jid, 1])
        mujoco.mj_forward(self.model, self.data)

    def test_free_bodies_no_grip_constraints_or_cosmetic_handles(self):
        self.assertEqual(self.model.neq, 0)
        self.assertEqual(sum(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 2)
        self.assertEqual(self.model.nu, 22)
        names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self.assertFalse(any("grip_handle" in name for name in names))

    def test_box_starts_above_both_forearms_without_penetration(self):
        box = self.model.geom("payload/payload_box").id
        distances = {}
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name
            if not name.startswith("robot/") or "collision" not in name:
                continue
            distance = mujoco.mj_geomDistance(self.model, self.data, gid, box, 1.0, None)
            distances[name] = distance
            self.assertGreater(distance, -1e-5, f"Initial penetration: {name}: {distance}")
        for side in ("left", "right"):
            self.assertLess(abs(distances[f"robot/{side}_hand_collision"]), 2e-4)
        self.assertGreater(self.data.qpos[self.payload_adr + 2], .5)

    def test_separated_payload_is_in_free_fall_not_pulled_towards_hands(self):
        self.data.qpos[self.payload_adr] += 1.0
        self.data.qpos[self.payload_adr + 2] += .3
        mujoco.mj_forward(self.model, self.data)
        np.testing.assert_allclose(self.data.qacc[self.payload_dof:self.payload_dof+3],
                                   [0, 0, -9.81], atol=1e-8)
        np.testing.assert_allclose(self.data.qacc[self.payload_dof+3:self.payload_dof+6],
                                   0, atol=1e-8)

    def test_forearm_configuration_does_not_reset_on_legacy_grip_offset(self):
        cfg = make_env_cfg(num_envs=1, support_mode="forearms")
        self.assertIsNone(cfg.scene.spec_fn)
        self.assertNotIn("grip_failed", cfg.terminations)
        self.assertIn("fell", cfg.terminations)
        self.assertEqual(cfg.support_mode, "forearms")
        self.assertIn("support_lost", cfg.terminations)
        self.assertIn("payload_disallowed_contact", cfg.terminations)
        self.assertIn("support_history", cfg.events)
        self.assertEqual(cfg.support_grace_steps, 6)

    def test_all_robot_colliders_are_named_and_sensor_exclusions_are_explicit(self):
        from lifting_lab.env_cfg import FOREARM_SUPPORT_GEOMS, ALLOWED_AUXILIARY_GEOMS
        cfg = make_env_cfg(num_envs=1, support_mode="forearms")
        forbidden = next(s for s in cfg.scene.sensors if s.name == "forbidden_robot_payload_contact")
        self.assertEqual(set(forbidden.primary.exclude),
                         set(FOREARM_SUPPORT_GEOMS + ALLOWED_AUXILIARY_GEOMS))
        for gid in range(self.model.ngeom):
            if self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]:
                self.assertTrue(self.model.geom(gid).name)

    def test_native_netforce_points_from_primary_floor_towards_secondary_box(self):
        # Independent minimal real-physics oracle for the sign and frame used
        # in mdp.forearm_contact_metrics (not a torch fixture assumption).
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, .1))
        body = spec.worldbody.add_body(name="load", pos=(0, 0, .049))
        body.add_freejoint()
        body.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=(.05, .05, .05), mass=1)
        spec.add_sensor(name="support", type=mujoco.mjtSensor.mjSENS_CONTACT,
                        objtype=mujoco.mjtObj.mjOBJ_GEOM, objname="floor",
                        reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="box",
                        intprm=(3, 3, 1))  # found + force, netforce, one slot
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.sensordata[0], 0)
        self.assertGreater(data.sensordata[3], 0)
        np.testing.assert_allclose(data.sensordata[1:3], 0, atol=1e-9)

    def test_invalid_mode_is_not_silently_treated_as_pregrasp(self):
        with self.assertRaises(ValueError):
            make_env_cfg(support_mode="unknown")


@unittest.skipUnless(os.environ.get("LIFTING_TEST_ENV") == "1", "Enable bounded CPU world checks")
class ForearmWorldTests(unittest.TestCase):
    def test_viewer_can_continue_after_box_drop_without_terminal_step_error(self):
        import torch
        from lifting_lab import cli
        from lifting_lab.viewer import prepare_world
        args = cli.build_parser().parse_args([
            "demo", "--viewer", "--support-mode", "forearms", "--mass", "2",
            "--seconds", "4", "--threads", "2",
        ])
        env, _, _, _ = prepare_world(args, cli._env, cli._policy)
        try:
            self.assertFalse(env.cfg.auto_reset)
            self.assertNotIn("fell", env.cfg.terminations)
            for _ in range(100):
                obs, _, terminated, _, _ = env.step(torch.zeros((1, 22)))
                self.assertFalse(terminated.any())
                self.assertTrue(torch.isfinite(obs["actor"]).all())
            self.assertEqual(int(env.episode_length_buf[0]), 100)
        finally:
            env.close()

    def test_mass_rebuild_retains_free_contact_model_and_finite_steps(self):
        import torch
        from lifting_lab import cli, mdp
        from lifting_lab.viewer import prepare_world, world_snapshot
        for mass in (.5, 2.0, 0.0):
            args = cli.build_parser().parse_args([
                "demo", "--support-mode", "forearms", "--mass", str(mass),
                "--seconds", "2", "--threads", "2",
            ])
            env, mass_info, policy, wrapper = prepare_world(args, cli._env, cli._policy)
            try:
                snapshot = world_snapshot(env, mass_info, 0, True)
                self.assertEqual(env.sim.mj_model.neq, 0)
                self.assertEqual(snapshot["control_steps"], 0)
                self.assertAlmostEqual(snapshot["compiled_payload_mass_kg"], max(mass, .001))
                metrics = mdp.task_metrics(env)
                self.assertNotIn("grip_error_m", metrics)
                self.assertIn("support_reference_offset_m", metrics)
                for _ in range(4):
                    obs, reward, _, _, _ = env.step(torch.zeros((1, 22)))
                    self.assertTrue(torch.isfinite(obs["actor"]).all())
                    self.assertTrue(torch.isfinite(reward).all())
            finally:
                env.close()


@unittest.skipUnless(os.environ.get("LIFTING_TEST_ENV") == "1", "Enable bounded CPU world checks")
class ForearmTrainingContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        torch.set_num_threads(2)
        cfg = make_env_cfg(num_envs=2, support_mode="forearms", payload_mass_kg=.5,
                           reset_xy_range_m=0, reset_yaw_range_rad=0, reset_speed_range_m_s=0)
        cfg.auto_reset = False
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=7)

    def test_real_contact_forces_and_auxiliary_hand_ball_contact(self):
        import torch
        from lifting_lab import mdp
        obs, reward, done, _, _ = self.env.step(torch.zeros(2, 22))
        values = mdp.task_metrics(self.env)
        self.assertEqual(tuple(obs["actor"].shape), (2, 97))
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(values["payload_forearm_supported"].all())
        self.assertTrue((values["forearm_support_force_n"] > 0).all())
        self.assertFalse(done.any())
        self.env.step(torch.zeros(2, 22))
        values = mdp.task_metrics(self.env)
        # Native end-ball contacts can carry load during startup; those are
        # allowed auxiliaries but do not qualify as cylinder support.
        self.assertFalse(values["payload_disallowed_contact"].any())
        self.assertTrue((values["support_gap_steps"] <= 6).all())

    def test_detached_payload_fails_on_seventh_control_step_without_hidden_force(self):
        import torch
        from lifting_lab import mdp
        payload = self.env.scene["payload"]
        state = payload.data.default_root_state.clone()
        state[:, :3] += self.env.scene.env_origins
        state[:, 0] += 1
        state[:, 2] += .4
        payload.write_root_state_to_sim(state)
        self.env.sim.forward()
        for step in range(1, 8):
            _, _, done, _, _ = self.env.step(torch.zeros(2, 22))
            values = mdp.task_metrics(self.env)
            self.assertEqual(values["support_gap_steps"].tolist(), [step, step])
            self.assertEqual(done.tolist(), [step == 7, step == 7])
            self.assertFalse(values["payload_disallowed_contact"].any())
        self.assertTrue((payload.data.root_link_lin_vel_w[:, 2] < -1).all())

    def test_box_on_floor_is_forbidden_not_successful_forearm_support(self):
        import torch
        from lifting_lab import mdp
        payload = self.env.scene["payload"]
        state = payload.data.default_root_state.clone()
        state[:, :3] += self.env.scene.env_origins
        state[:, 0] += 1
        state[:, 2] = self.env.scene.env_origins[:, 2] + .059
        payload.write_root_state_to_sim(state)
        self.env.sim.forward()
        _, _, done, _, _ = self.env.step(torch.zeros(2, 22))
        values = mdp.task_metrics(self.env)
        self.assertTrue(values["payload_disallowed_contact"].all())
        self.assertFalse(values["payload_forearm_supported"].any())
        self.assertTrue(done.all())
        self.assertFalse(values["valid_hold"].any())

    def test_torso_contact_is_forbidden_and_isolated_to_that_environment(self):
        import torch
        from lifting_lab import mdp
        payload = self.env.scene["payload"]
        state = payload.data.default_root_state[:1].clone()
        state[:, :3] += self.env.scene.env_origins[:1]
        state[:, 0] = self.env.scene.env_origins[:1, 0]
        payload.write_root_state_to_sim(state, torch.tensor([0]))
        self.env.sim.forward()
        self.env.step(torch.zeros(2, 22))
        forbidden = mdp.task_metrics(self.env)["payload_disallowed_contact"]
        self.assertEqual(forbidden.tolist(), [True, False])

    def test_partial_reset_only_clears_selected_support_history(self):
        import torch
        from lifting_lab import mdp
        self.env.step(torch.zeros(2, 22))
        self.env.step(torch.zeros(2, 22))
        old = mdp.forearm_contact_metrics(self.env)["support_gap_steps"].clone()
        self.env.reset(env_ids=torch.tensor([0]))
        values = mdp.forearm_contact_metrics(self.env)
        self.assertEqual(int(values["support_gap_steps"][0]), 0)
        self.assertEqual(values["support_gap_steps"][1], old[1])


if __name__ == "__main__":
    unittest.main()
