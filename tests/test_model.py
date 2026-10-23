"""Native MuJoCo model tests: real inertia and dynamics, not metadata checks.

No Warp simulator, renderer, training run, or CUDA device is started here.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab.model import (
    GRIP_SITES, HAND_SITES, MIN_PAYLOAD_MASS_KG, PAYLOAD_BODY,
    PayloadCfg, add_grip_constraints, holding_keyframe, initial_grip_geometry,
    known_base_robot_mass, make_payload_cfg, make_payload_spec, make_robot_cfg,
)


def isolated_payload(mass):
    model = make_payload_spec(PayloadCfg(mass_kg=mass)).compile()
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class PayloadModelTests(unittest.TestCase):
    def test_invalid_mass_is_rejected(self):
        for mass in (-1.0, np.nan, np.inf, -np.inf):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                PayloadCfg(mass_kg=mass)

    def test_invalid_box_dimensions_are_rejected(self):
        for size in ((0.1, 0.2), (0.1, 0.2, 0), (0.1, -0.2, 0.1),
                     (0.1, np.nan, 0.1), (0.1, np.inf, 0.1)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                PayloadCfg(half_size=size)

    def test_requested_zero_is_explicit_one_gram_floor(self):
        cfg = PayloadCfg(mass_kg=0)
        self.assertEqual(cfg.mass_kg, 0)
        self.assertEqual(cfg.actual_mass_kg, MIN_PAYLOAD_MASS_KG)
        self.assertEqual(MIN_PAYLOAD_MASS_KG, 0.001)
        model = make_payload_spec(cfg).compile()
        self.assertAlmostEqual(model.body_mass[model.body(PAYLOAD_BODY).id], 0.001)
        self.assertTrue(np.all(model.body_inertia[model.body(PAYLOAD_BODY).id] > 0))

    def test_homogeneous_box_inertia_and_mass_are_in_compiled_model(self):
        cfg = PayloadCfg(mass_kg=2.0, half_size=(0.1, 0.2, 0.06))
        # For half dimensions h, Ixx = m / 3 * (hy**2 + hz**2).
        expected = 2.0 / 3.0 * np.array([0.2**2 + 0.06**2,
                                       0.1**2 + 0.06**2,
                                       0.1**2 + 0.2**2])
        np.testing.assert_allclose(cfg.diagonal_inertia, expected, rtol=1e-12)
        model = make_payload_spec(cfg).compile()
        body_id = model.body(PAYLOAD_BODY).id
        self.assertAlmostEqual(model.body_mass.sum(), 2.0)
        np.testing.assert_allclose(model.body_inertia[body_id], expected, rtol=1e-12)

    def test_payload_remains_a_free_six_dof_rigid_body(self):
        model, _ = isolated_payload(1.0)
        self.assertEqual(model.nq, 7)
        self.assertEqual(model.nv, 6)
        self.assertEqual(model.nu, 0)
        self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)

    def test_mass_changes_generalized_inertia_matrix(self):
        matrices = []
        for mass in (0.5, 2.0):
            model, data = isolated_payload(mass)
            dense = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, data, dense)
            self.assertTrue(np.all(np.linalg.eigvalsh(dense) > 0))
            matrices.append(dense)
        np.testing.assert_allclose(matrices[1], 4.0 * matrices[0], rtol=1e-12)

    def test_equal_external_force_produces_inverse_mass_acceleration(self):
        accelerations = []
        for mass in (0.5, 2.0):
            model, data = isolated_payload(mass)
            body_id = model.body(PAYLOAD_BODY).id
            data.xfrc_applied[body_id, 0] = 8.0  # World-x force at body COM, N.
            mujoco.mj_forward(model, data)
            accelerations.append(float(data.qacc[0]))
            self.assertAlmostEqual(data.qacc[0], 8.0 / mass, places=8)
            np.testing.assert_allclose(data.qacc[1:], 0, atol=1e-10)
        self.assertAlmostEqual(accelerations[0] / accelerations[1], 4.0, places=8)

    def test_equal_torque_produces_inverse_inertia_angular_acceleration(self):
        for mass in (0.5, 2.0):
            model, data = isolated_payload(mass)
            body_id = model.body(PAYLOAD_BODY).id
            data.xfrc_applied[body_id, 3] = 0.1
            mujoco.mj_forward(model, data)
            expected = 0.1 / PayloadCfg(mass_kg=mass).diagonal_inertia[0]
            self.assertAlmostEqual(data.qacc[3], expected, places=8)

    def test_actual_integrated_motion_depends_on_mass(self):
        positions = []
        for mass in (0.5, 2.0):
            model, data = isolated_payload(mass)
            model.opt.timestep = 0.001
            body_id = model.body(PAYLOAD_BODY).id
            data.xfrc_applied[body_id, 0] = 8.0
            for _ in range(20):
                mujoco.mj_step(model, data)
            self.assertTrue(np.isfinite(data.qpos).all())
            positions.append(float(data.qpos[0]))
        self.assertGreater(positions[0], 0)
        self.assertAlmostEqual(positions[0] / positions[1], 4.0, places=8)

    def test_robot_mass_is_finite_positive_and_independent_of_payload(self):
        before = known_base_robot_mass()
        make_payload_spec(PayloadCfg(mass_kg=10.0)).compile()
        self.assertTrue(np.isfinite(before))
        self.assertGreater(before, 1.0)
        self.assertEqual(known_base_robot_mass(), before)

    def test_initial_state_is_returned_as_an_independent_copy(self):
        changed = holding_keyframe()
        original = holding_keyframe()
        changed.joint_pos["Left_Elbow_Pitch"] += 0.1
        self.assertEqual(holding_keyframe().joint_pos, original.joint_pos)


class PregraspModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.scene import Scene, SceneCfg

        cls.scene = Scene(SceneCfg(num_envs=1, entities={
            "robot": make_robot_cfg(),
            "payload": make_payload_cfg(PayloadCfg(mass_kg=1.0)),
        }, spec_fn=add_grip_constraints), device="cpu")
        cls.model = cls.scene.compile()
        cls.data = mujoco.MjData(cls.model)
        robot_state = holding_keyframe()
        for joint_id in range(cls.model.njnt):
            name = cls.model.joint(joint_id).name
            qadr = cls.model.jnt_qposadr[joint_id]
            if cls.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                if name.startswith("robot/"):
                    cls.data.qpos[qadr:qadr + 7] = (*robot_state.pos, *robot_state.rot)
                else:
                    center, _ = initial_grip_geometry()
                    cls.data.qpos[qadr:qadr + 7] = (*center, 1, 0, 0, 0)
            elif name.startswith("robot/"):
                cls.data.qpos[qadr] = robot_state.joint_pos.get(name.split("/", 1)[1], 0.0)
        mujoco.mj_forward(cls.model, cls.data)

    def test_robot_and_payload_are_not_welded_to_world(self):
        self.assertEqual(np.count_nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 2)
        self.assertEqual(np.count_nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE), 22)
        self.assertEqual(self.model.nu, 22)

    def test_two_site_connections_start_with_negligible_position_error(self):
        self.assertEqual(self.model.neq, 2)
        np.testing.assert_array_equal(self.model.eq_type, mujoco.mjtEq.mjEQ_CONNECT)
        np.testing.assert_array_equal(self.model.eq_objtype, mujoco.mjtObj.mjOBJ_SITE)
        for hand, grip in zip(HAND_SITES, GRIP_SITES, strict=True):
            hand_id = self.model.site(f"robot/{hand}").id
            grip_id = self.model.site(f"payload/{grip}").id
            residual = np.linalg.norm(self.data.site_xpos[hand_id] - self.data.site_xpos[grip_id])
            self.assertLess(residual, 1e-7, f"Initial pregrasp mismatch: {hand}: {residual} m")


if __name__ == "__main__":
    unittest.main()
