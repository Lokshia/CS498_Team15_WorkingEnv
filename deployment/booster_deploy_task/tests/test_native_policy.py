"""Offline native-policy contract tests; set TEAM15_UPSTREAM_ROOT to booster_deploy."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = os.environ.get("TEAM15_UPSTREAM_ROOT")
if UPSTREAM:
    sys.path.insert(0, UPSTREAM)
sys.path.insert(0, str(HERE.parent/"tasks"))


class NativePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_module = importlib.import_module("team15_hold.policy")
        cls.constants = importlib.import_module("team15_hold.contract")
        cls.task_module = importlib.import_module("team15_hold.task")
        cls.check_module = importlib.import_module("team15_hold.check")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"TEAM15_LOG_DIR":self.temp.name})
        self.env.start()
        cfg = self.task_module.Team15HoldControllerCfg()
        self.controller = self.check_module.OfflineController(cfg)
        self.policy = self.policy_module.Team15HoldPolicy(cfg.policy, self.controller)
        self.controller.robot.data.joint_pos.copy_(self.policy.nominal)
        self.controller.robot.data.root_quat_w.copy_(torch.tensor([1.,0.,0.,0.]))

    def tearDown(self):
        self.policy.close()
        self.env.stop()
        self.temp.cleanup()

    def test_full_export_oracle_and_config(self):
        report = self.check_module.check()
        self.assertEqual(report["test_vectors"], 534)
        self.assertTrue(report["offline_check_passed"])
        self.assertFalse(report["hardware_tested"])
        self.assertNotIn("rclpy", sys.modules)

    def test_observation_and_feedback_are_exact(self):
        self.policy.reset()
        data = self.controller.robot.data
        data.root_ang_vel_b.copy_(torch.tensor([.1,.2,.3]))
        data.joint_vel.copy_(torch.arange(22)*.01)
        raw = torch.linspace(-1.,1.,22)
        targets = self.policy.action_to_targets(raw)
        expected_targets = np.clip(np.asarray(self.constants.DEFAULT_JOINT_POS,dtype=np.float32)+np.asarray(self.constants.ACTION_SCALE,dtype=np.float32)*raw.numpy(),
                                   np.asarray(self.constants.TARGET_RANGES_RAD,dtype=np.float32)[:,0], np.asarray(self.constants.TARGET_RANGES_RAD,dtype=np.float32)[:,1])
        np.testing.assert_array_equal(targets.numpy(), expected_targets)
        obs = self.policy.compute_observation().numpy()
        np.testing.assert_array_equal(obs[:3], data.root_ang_vel_b.numpy())
        np.testing.assert_array_equal(obs[3:6], [0,0,-1])
        np.testing.assert_array_equal(obs[6:28], np.zeros(22))
        np.testing.assert_array_equal(obs[28:50], data.joint_vel.numpy())
        np.testing.assert_array_equal(obs[50:72], self.policy.last_action.numpy())
        changed = expected_targets != (self.policy.nominal+self.policy.scale*raw).numpy()
        np.testing.assert_array_equal(self.policy.last_action.numpy()[~changed], raw.numpy()[~changed])
        np.testing.assert_array_equal(self.policy.last_action.numpy()[changed], ((targets-self.policy.nominal)/self.policy.scale).numpy()[changed])

    def test_reset_rejects_bad_state_and_pose_without_returning_target(self):
        for kind in ("nan", "quaternion", "pose", "tilt"):
            with self.subTest(kind=kind):
                data = self.controller.robot.data
                data.joint_pos.copy_(self.policy.nominal)
                data.root_quat_w.copy_(torch.tensor([1.,0.,0.,0.]))
                data.root_ang_vel_b.zero_()
                self.controller.stopped = False
                if kind == "nan": data.root_ang_vel_b[0] = float("nan")
                if kind == "quaternion": data.root_quat_w.zero_()
                if kind == "pose": data.joint_pos[3] += .5
                if kind == "tilt": data.root_quat_w.copy_(torch.tensor([.70710678,.70710678,0.,0.]))
                with self.assertRaises((ValueError,RuntimeError)):
                    self.policy.reset()
                self.assertTrue(self.controller.stopped)
                self.assertFalse(self.policy._ready)

    def test_inference_fault_stops_then_raises(self):
        self.policy.reset()
        self.controller.robot.data.root_ang_vel_b[0] = float("nan")
        with self.assertRaises(ValueError):
            self.policy.inference()
        self.assertTrue(self.controller.stopped)
        self.assertFalse(self.policy._ready)

    def test_log_close_error_still_stops_controller(self):
        self.policy.reset()
        self.controller.robot.data.root_ang_vel_b[0] = float("nan")
        with patch.object(self.policy, "close", side_effect=OSError("simulated log close failure")):
            with self.assertRaises(OSError):
                self.policy.inference()
        self.assertTrue(self.controller.stopped)
        self.assertFalse(self.policy._ready)

    def test_logging_unique_full_step_and_no_ack_claim(self):
        self.policy.reset()
        first = self.policy.log_path
        target = self.policy.inference()
        self.policy.reset()
        second = self.policy.log_path
        self.assertNotEqual(first,second)
        rows = [json.loads(row) for row in first.read_text().splitlines()]
        self.assertEqual(len(rows),2)
        self.assertIn("not publish",rows[0]["log_semantics"])
        self.assertFalse(rows[0]["state_receive_age_available"])
        self.assertEqual(len(rows[0]["run_gain_sha256"]),64)
        self.assertEqual(rows[0]["task_configuration"]["prepare_mode"],"standing")
        for row in rows:
            self.assertIsInstance(row["wall_time_ns"],int)
            self.assertIn("monotonic_s",row)
        step = rows[1]
        self.assertEqual(len(step["observation"]),72)
        self.assertEqual(len(step["raw_action"]),22)
        np.testing.assert_array_equal(step["target_controller_order_rad"],target.numpy())

    def test_joint_order_mapping(self):
        # Exercise a nonidentity controller order; target and observation must
        # map by name, not assume left/right/serial indices are interchangeable.
        cfg = self.task_module.Team15HoldControllerCfg()
        cfg.robot.joint_names = cfg.robot.joint_names[::-1]
        cfg.robot.default_joint_pos = cfg.robot.default_joint_pos[::-1]
        c = self.check_module.OfflineController(cfg)
        p = self.policy_module.Team15HoldPolicy(cfg.policy,c)
        try:
            c.robot.data.joint_pos.copy_(torch.flip(p.nominal,[0]))
            c.robot.data.root_quat_w.copy_(torch.tensor([1.,0.,0.,0.]))
            p.reset()
            torch.testing.assert_close(p.compute_observation()[6:28],torch.zeros(22),atol=0,rtol=0)
            torch.testing.assert_close(p.action_to_targets(torch.zeros(22)),torch.flip(p.nominal,[0]),atol=0,rtol=0)
        finally:
            p.close()

    def test_projection_disabled_is_rejected(self):
        cfg = self.task_module.Team15HoldControllerCfg()
        cfg.policy.project_joint_targets = False
        with self.assertRaises(ValueError):
            self.policy_module.Team15HoldPolicy(cfg.policy,self.check_module.OfflineController(cfg))

    def test_frozen_envelope_matches_existing_validation_evidence(self):
        old = HERE.parents[1]/"team15_proprio_package/reports/deploy_0p5kg/report.json"
        if not old.exists():
            self.skipTest("Original evidence is outside this stand-alone task package")
        report = json.loads(old.read_text(encoding="utf-8"))
        np.testing.assert_array_equal(report["model_audit"]["joint_ranges_rad"],self.constants.TARGET_RANGES_RAD)


if __name__ == "__main__":
    unittest.main()
