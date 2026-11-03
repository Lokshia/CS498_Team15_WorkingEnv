"""Portable regression checks for the free payload's inertial frame.

From test_workspace (or use absolute paths):
  python team15_proprio_package/tests/test_payload_inertia.py \
      --deploy-root booster_deploy --assets-root booster_assets -v

Requires the package task installed in booster_deploy and its ordinary simulation
dependencies. No training code, mjlab, GPU, viewer, ROS or robot connection is used.
These three tests check the scene; they do not replace validate_package.py's
closed-loop policy evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


class PayloadInertiaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.initial, cls.audit = cls.scene.build_scene(cls.asset_xml, .5)

    def test_center_of_mass_and_inertial_orientation(self):
        model = self.model
        data = mujoco.MjData(model)
        data.qpos[:] = self.initial
        mujoco.mj_forward(model, data)
        body = model.body(self.scene.PAYLOAD_BODY).id
        geom = model.geom(self.scene.PAYLOAD_GEOM).id
        np.testing.assert_allclose(model.body_ipos[body], [0., 0., 0.], rtol=0., atol=1e-12)
        np.testing.assert_allclose(model.body_iquat[body], [1., 0., 0., 0.], rtol=0., atol=1e-12)
        np.testing.assert_allclose(data.xipos[body], data.geom_xpos[geom], rtol=0., atol=1e-12)

    def test_uniform_box_mass_and_diagonal_inertia(self):
        for requested_mass in (.25, .5, 1.):
            with self.subTest(mass_kg=requested_mass):
                model = self.model if requested_mass == .5 else self.scene.build_scene(
                    self.asset_xml, requested_mass)[0]
                body = model.body(self.scene.PAYLOAD_BODY).id
                geom = model.geom(self.scene.PAYLOAD_GEOM).id
                self.assertEqual(model.geom_type[geom], mujoco.mjtGeom.mjGEOM_BOX)
                self.assertAlmostEqual(float(model.body_mass[body]), requested_mass, places=12)
                x, y, z = model.geom_size[geom]
                expected = requested_mass*np.array([y*y+z*z, x*x+z*z, x*x+y*y])/3
                self.assertTrue(np.isfinite(model.body_inertia[body]).all())
                self.assertTrue((model.body_inertia[body] > 0).all())
                np.testing.assert_allclose(model.body_inertia[body], expected, rtol=1e-12, atol=1e-12)

    def test_force_through_geometric_center_produces_no_rotation(self):
        model = self.model
        data = mujoco.MjData(model)
        data.qpos[:] = self.initial
        body = model.body(self.scene.PAYLOAD_BODY).id
        geom = model.geom(self.scene.PAYLOAD_GEOM).id
        joint = model.joint(self.scene.PAYLOAD_JOINT).id
        self.assertEqual(model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.neq, 0)
        qadr, vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        # Isolate the payload from every support/ground contact before applying
        # a force at its visible center, not at the possibly incorrect COM.
        data.qpos[qadr:qadr+3] += [2., 0., 1.]
        mujoco.mj_forward(model, data)
        self.assertFalse(any(geom in (c.geom1, c.geom2) for c in data.contact))
        force = np.array([.2, -.3, 1.])
        mujoco.mj_applyFT(model, data, force, np.zeros(3), data.geom_xpos[geom].copy(),
                         body, data.qfrc_applied)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qacc[vadr:vadr+3],
            model.opt.gravity+force/model.body_mass[body], rtol=0., atol=1e-9)
        np.testing.assert_allclose(data.qacc[vadr+3:vadr+6], 0., rtol=0., atol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy-root", type=Path, required=True, help="booster_deploy checkout with installed Team15 task")
    parser.add_argument("--assets-root", type=Path, required=True, help="booster_assets checkout")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    deploy = args.deploy_root.resolve(strict=True)
    assets = args.assets_root.resolve(strict=True)
    sys.path.insert(0, str(assets/"src"))
    sys.path.insert(0, str(deploy))
    import booster_assets
    from tasks.team15_proprio import scene
    # Refuse to silently test a task or asset installation from a different path.
    Path(scene.__file__).resolve(strict=True).relative_to(deploy/"tasks/team15_proprio")
    asset_xml = (Path(booster_assets.BOOSTER_ASSETS_DIR)/"robots/K1/K1_22dof.xml").resolve(strict=True)
    asset_xml.relative_to(assets)
    PayloadInertiaTests.scene = scene
    PayloadInertiaTests.asset_xml = asset_xml
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PayloadInertiaTests)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
