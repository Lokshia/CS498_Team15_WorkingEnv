"""Independent controller math/limits tests and real exported-actor parity.

All state/control fixtures here are synthetic. No ROS imports, robot interface,
or approved hardware parameter claims are involved.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
sys.path.insert(0, str(BRIDGE))
sys.path.insert(0, str(HERE))
from fixtures import make_config
from team15_hardware.core import Actor, ControlCore, SafetyFault, prepare_target, world_from_imu
from team15_hardware.ros_transport import StateSnapshot

MODEL_DIR = BRIDGE.parent / "team15_proprio_package/tasks/team15_proprio/models"


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((MODEL_DIR / "proprio_hold.contract.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.config = make_config(self.contract)
        self.config['prepare']['start_model_q'] = self.contract['joint_mapping']['default_joint_pos_rad'].copy()
        self.config['limits'].update(state_timeout_s=.1,command_timeout_s=.1,max_trunk_tilt_deg=30.)
        self.core = ControlCore(self.config,self.contract)

    def snapshot(self, q=None, dq=None, tau=None, *, received=100., sequence=1, rpy=(0.,0.,0.), gyro=(0.,0.,0.)):
        q = self.core.nominal if q is None else np.asarray(q,dtype=float)
        dq = np.zeros(22) if dq is None else np.asarray(dq,dtype=float)
        tau = np.zeros(22) if tau is None else np.asarray(tau,dtype=float)
        # Build raw serial readings independently of ControlCore conversion.
        raw_q,raw_dq,raw_tau = np.empty(22),np.empty(22),np.empty(22)
        for i,row in enumerate(self.config['joints']):
            j=row['serial_index']; s=row['sign']; offset=row['zero_offset_rad']
            raw_q[j]=offset+s*q[i]; raw_dq[j]=s*dq[i]; raw_tau[j]=s*tau[i]
        return StateSnapshot(tuple(raw_q),tuple(raw_dq),tuple(raw_tau),tuple(rpy),tuple(gyro),received,sequence)

    def prime(self):
        snapshot=self.snapshot()
        command=self.core.prime(snapshot,100.)
        self.core.commit_sent(command,100.)
        return snapshot

    def test_permuted_signed_offset_joint_mapping_round_trip(self):
        for i,row in enumerate(self.config['joints']):
            row.update(serial_index=21-i,sign=-1 if i%2 else 1,zero_offset_rad=i*.003)
        self.core=ControlCore(self.config,self.contract)
        q=self.core.nominal+np.linspace(-.01,.01,22)
        dq=np.linspace(-.02,.02,22)
        snapshot=self.snapshot(q=q,dq=dq)
        actual_q,actual_dq,*_=self.core.state(snapshot,100.)
        np.testing.assert_allclose(actual_q,q,atol=1e-12)
        np.testing.assert_allclose(actual_dq,dq,atol=1e-12)
        np.testing.assert_allclose(self.core.to_serial(q),snapshot.q,atol=1e-12)
        # Gains are permuted but never multiplied by a joint sign.
        np.testing.assert_allclose(self.core.kp[::-1],[r['kp'] for r in self.config['joints']])

    def test_nonidentity_imu_rotation_transforms_gyro_and_gravity(self):
        self.config['imu']['body_from_imu']=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        self.config['limits']['max_trunk_tilt_deg']=179.
        self.core=ControlCore(self.config,self.contract)
        snapshot=self.snapshot(rpy=(np.pi/2,0.,0.),gyro=(1.,2.,3.))
        _,_,gravity,gyro=self.core.state(snapshot,100.)
        np.testing.assert_allclose(gravity,[1.,0.,0.],atol=1e-12)
        np.testing.assert_allclose(gyro,[-2.,1.,3.],atol=1e-12)

    def test_imu_rotation_is_orthogonal_and_yaw_does_not_change_vertical(self):
        rotation=world_from_imu([.2,-.3,.8])
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(world_from_imu([0.,0.,1.2]).T@[0.,0.,-1.],[0.,0.,-1.],atol=1e-12)

    def test_observation_order_and_zero_initial_previous_action(self):
        q=self.core.nominal+np.linspace(-.01,.01,22)
        dq=np.linspace(-.02,.02,22)
        obs=self.core.observation(self.snapshot(q=q,dq=dq,gyro=(.1,.2,.3)),100.)
        self.assertEqual(obs.shape,(72,))
        self.assertEqual(obs.dtype,np.float32)
        np.testing.assert_allclose(obs[:3],[.1,.2,.3])
        np.testing.assert_allclose(obs[3:6],[0.,0.,-1.])
        np.testing.assert_allclose(obs[6:28],q-self.core.nominal,atol=1e-8)
        np.testing.assert_allclose(obs[28:50],dq,atol=1e-8)
        np.testing.assert_array_equal(obs[50:],np.zeros(22))

    def test_stale_future_and_nonfinite_state_rejected(self):
        for received in (99.,101.,float('nan')):
            with self.subTest(received=received),self.assertRaises(SafetyFault):
                self.core.state(self.snapshot(received=received),100.)
        for field,n in (('q',22),('dq',22),('tau',22),('rpy',3),('gyro',3)):
            snapshot=replace(self.snapshot(),**{field:(float('nan'),)+(0.,)*(n-1)})
            with self.subTest(field=field),self.assertRaises(SafetyFault):
                self.core.state(snapshot,100.)

    def test_state_joint_speed_torque_position_and_tilt_limits(self):
        q=self.core.nominal.copy(); q[0]=self.core.upper[0]+.01
        dq=np.zeros(22); dq[0]=self.core.max_speed[0]+.01
        tau=np.zeros(22); tau[0]=self.core.max_torque[0]+.01
        cases=[self.snapshot(q=q),self.snapshot(dq=dq),self.snapshot(tau=tau),self.snapshot(rpy=(1.,0.,0.))]
        for snapshot in cases:
            with self.assertRaises(SafetyFault): self.core.state(snapshot,100.)

    def test_prime_requires_reviewed_starting_pose(self):
        self.config['prepare']['start_tolerance_rad']=.005
        q=self.core.nominal.copy(); q[0]+=.01
        with self.assertRaisesRegex(SafetyFault,'starting pose'):
            self.core.prime(self.snapshot(q=q),100.)

    def test_no_target_without_successful_prime_publication(self):
        with self.assertRaisesRegex(SafetyFault,'published starting'):
            self.core.policy_command(np.ones(22),self.snapshot(),100.)

    def test_targets_rate_limited_and_previous_action_committed_only_after_send(self):
        for row in self.config['joints']: row['target_rate_limit_rad_s']=.5
        self.core=ControlCore(self.config,self.contract)
        self.prime()
        snapshot=self.snapshot(received=100.02,sequence=2)
        command=self.core.policy_command(np.ones(22),snapshot,100.02)
        np.testing.assert_allclose(command.model_q,self.core.nominal+.01,atol=1e-12)
        np.testing.assert_array_equal(self.core.previous_action,np.zeros(22))
        np.testing.assert_array_equal(self.core.previous_target,self.core.nominal)
        self.core.commit_sent(command,100.021)
        np.testing.assert_allclose(self.core.previous_action,.01/self.core.scale,atol=1e-12)
        obs=self.core.observation(snapshot,100.021)
        np.testing.assert_allclose(obs[50:],command.effective_action,atol=1e-8)
        self.assertEqual(command.observation_sequence,2)

    def test_position_projection_is_reflected_in_previous_action(self):
        for row in self.config['joints']: row['target_rate_limit_rad_s']=5.
        self.config['joints'][0]['model_max_rad']=self.core.nominal[0]+.005
        self.core=ControlCore(self.config,self.contract)
        self.prime()
        command=self.core.policy_command(np.ones(22),self.snapshot(received=100.02),100.02)
        self.assertAlmostEqual(command.model_q[0],self.core.nominal[0]+.005)
        self.core.commit_sent(command,100.02)
        self.assertAlmostEqual(self.core.previous_action[0],.005/self.core.scale[0])

    def test_late_loop_never_allows_larger_target_jump(self):
        for row in self.config['joints']: row['target_rate_limit_rad_s']=.5
        self.core=ControlCore(self.config,self.contract)
        self.prime()
        command=self.core.policy_command(np.ones(22),self.snapshot(received=100.06),100.06)
        np.testing.assert_allclose(command.model_q-self.core.nominal,.01,atol=1e-12)
        with self.assertRaisesRegex(SafetyFault,'missed'):
            self.core.policy_command(np.ones(22),self.snapshot(received=100.2),100.2)

    def test_tracking_error_and_nonfinite_action_stop_before_command(self):
        self.config['joints'][0]['tracking_error_limit_rad']=1e-5
        self.core=ControlCore(self.config,self.contract)
        self.prime()
        with self.assertRaisesRegex(SafetyFault,'tracking error'):
            self.core.policy_command(np.ones(22),self.snapshot(received=100.02),100.02)
        with self.assertRaises(SafetyFault):
            self.core.policy_command(np.full(22,np.nan),self.snapshot(received=100.02),100.02)

    def test_invalid_send_timestamp_does_not_commit_previous_action(self):
        command=self.core.prime(self.snapshot(),100.)
        for sent in (float('nan'),float('inf'),99.,101.):
            with self.subTest(sent=sent),self.assertRaises(SafetyFault):
                self.core.commit_sent(command,sent)
        self.assertIsNone(self.core.previous_target)
        np.testing.assert_array_equal(self.core.previous_action,np.zeros(22))

    def test_preparation_smoothstep_waypoints_and_completion(self):
        start=self.core.nominal.copy(); end=start+.01
        self.config['prepare'].update(start_model_q=start.tolist(),waypoints=[{'model_q':end.tolist(),'duration_s':2.}])
        first,done=prepare_target(self.config,0.)
        np.testing.assert_allclose(first,start); self.assertFalse(done)
        mid,done=prepare_target(self.config,1.)
        np.testing.assert_allclose(mid,(start+end)/2); self.assertFalse(done)
        final,done=prepare_target(self.config,2.)
        np.testing.assert_allclose(final,end); self.assertTrue(done)


class ActorParityTests(unittest.TestCase):
    def test_actual_exported_onnx_all_534_oracle_vectors(self):
        actor=Actor(MODEL_DIR/'proprio_hold.onnx')
        report=actor.verify_vectors(MODEL_DIR/'proprio_hold.test_vectors.npz')
        self.assertEqual(report['test_vectors'],534)
        self.assertLess(report['max_abs_error'],2e-5)

    def test_modified_model_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'proprio_hold.onnx'
            path.write_bytes(b'not the reviewed model')
            with self.assertRaisesRegex(SafetyFault,'hash'):
                Actor(path)


if __name__=='__main__':
    unittest.main()
