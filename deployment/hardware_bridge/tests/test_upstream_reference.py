"""Source-backed prefill must stay distinct from measured robot calibration."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from fixtures import load_contract
from team15_hardware.config import ConfigurationError, validate_config


class UpstreamReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile_path = ROOT/'profiles/k1.upstream-reference.json'
        cls.profile = json.loads(cls.profile_path.read_text(encoding='utf-8'))
        cls.sources = json.loads((ROOT/'profiles/k1.upstream-reference.sources.json').read_text(encoding='utf-8'))

    def test_frozen_profile_matches_source_receipt(self):
        self.assertEqual(hashlib.sha256(self.profile_path.read_bytes()).hexdigest(), self.sources['profile_sha256'])
        self.assertEqual(self.sources['upstream_commit'], '7bb1462e2ac742eba5c01659b815e2e99fbe617f')
        self.assertEqual(len(self.sources['source_files']), 3)
        for path, entry in self.sources['source_files'].items():
            self.assertEqual(len(entry['sha256']), 64)
            self.assertIn('/'+self.sources['upstream_commit']+'/'+path, entry['url'])
            self.assertEqual(entry['hash_basis'], 'raw Git blob bytes at pinned commit')

    def test_source_order_and_official_run_gains(self):
        expected_kp = [4.]*10 + [80.,80.,80.,80.,30.,30.]*2
        expected_kd = [1.]*10 + [2.]*12
        upstream = self.sources['upstream_k1_run_defaults']
        names = load_contract()['joint_mapping']['booster_deploy_joint_names']
        self.assertEqual(upstream['joint_names'], names)
        self.assertEqual(upstream['joint_stiffness'], expected_kp)
        self.assertEqual(upstream['joint_damping'], expected_kd)
        self.assertEqual([j['name'] for j in self.profile['joints']], names)
        self.assertEqual([j['serial_index'] for j in self.profile['joints']], list(range(22)))
        self.assertEqual([j['kp'] for j in self.profile['joints']], expected_kp)
        self.assertEqual([j['kd'] for j in self.profile['joints']], expected_kd)

    def test_upstream_assumptions_not_reported_as_robot_measurements(self):
        for row in self.profile['joints']:
            self.assertEqual(row['sign'], 1.)
            self.assertEqual(row['zero_offset_rad'], 0.)
            self.assertIsNone(row['measured_torque_limit_nm'])
        self.assertFalse(self.profile['_reference_status']['calibration_measured'])
        self.assertFalse(self.profile['_reference_status']['gains_validated_for_holding'])
        evidence = self.sources['robot_introspection']
        self.assertIsNone(evidence['actual_serial_motor_count'])
        self.assertFalse(evidence['serial_mapping_measured'])
        self.assertEqual(evidence['hardware_commands_sent'], 0)

    def test_prepare_defaults_are_separate_and_not_enabled(self):
        prepare = self.sources['upstream_k1_prepare_defaults']
        self.assertFalse(prepare['applied_to_new_bridge'])
        self.assertEqual(prepare['prepare_mode'], 'walking')
        self.assertEqual(prepare['stiffness'][10], 350.)
        self.assertNotEqual(prepare['stiffness'], [j['kp'] for j in self.profile['joints']])
        self.assertTrue(all(q is None for q in self.profile['prepare']['start_model_q']))

    def test_observed_interfaces_and_qos_recorded_without_private_endpoint_ids(self):
        evidence = self.sources['robot_introspection']
        self.assertEqual((evidence['machine'], evidence['python'], evidence['ros_distro']), ('aarch64','3.10.12','humble'))
        self.assertEqual(evidence['joint_ctrl']['publisher_count'], 0)
        self.assertEqual(evidence['joint_ctrl']['subscription_count'], 2)
        self.assertEqual(evidence['low_state']['publisher_qos']['reliability'], 'RELIABLE')
        self.assertEqual(evidence['joint_ctrl']['subscriber_qos']['reliability'], 'BEST_EFFORT')
        self.assertEqual(self.profile['transport']['low_state_topic'], '/low_state')
        self.assertEqual(self.profile['transport']['joint_ctrl_topic'], '/joint_ctrl')
        self.assertEqual(self.profile['transport']['rpc_service'], '/booster_rpc_service')
        self.assertNotIn('GID', json.dumps(evidence))

    def test_incomplete_reference_is_still_rejected_by_live_preflight(self):
        self.assertTrue(all(v is None for v in self.profile['review'].values()))
        with self.assertRaises(ConfigurationError) as context:
            validate_config(self.profile, load_contract())
        text = str(context.exception)
        for field in ('review.calibration_verified', 'limits.command_timeout_s',
                      'prepare.start_model_q', 'stop.firmware_watchdog_timeout_s'):
            self.assertIn(field, text)
        for field in ('transport.low_state_topic', 'joints[0].serial_index', 'joints[0].kp', 'joints[0].kd'):
            self.assertNotIn(field+': missing', text)


if __name__ == '__main__':
    unittest.main()
