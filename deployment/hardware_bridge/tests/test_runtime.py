"""Control-session fault injection; mocked transport, no ROS or robot access."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np

BRIDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE))
from team15_hardware.core import ControlCore, SafetyFault
from team15_hardware.ros_transport import StateSnapshot
from team15_hardware.runtime import Session


class Clock:
    def __init__(self): self.value = 100.
    def __call__(self): return self.value
    def advance(self, seconds=.02): self.value += seconds


def session_config(contract):
    """Deliberately synthetic fixture, never a robot profile."""
    nominal = contract['joint_mapping']['default_joint_pos_rad']
    return {
        'deployment_target': 'simulation_only',
        'review': {'robot_serial': 'SIM-ONLY'},
        'joints': [dict(serial_index=i, sign=1., zero_offset_rad=0., model_min_rad=-3.,
                        model_max_rad=3., target_rate_limit_rad_s=5., measured_speed_limit_rad_s=10.,
                        measured_torque_limit_nm=100., tracking_error_limit_rad=1., kp=1., kd=1.)
                   for i in range(22)],
        'imu': {'body_from_imu': np.eye(3).tolist()},
        'limits': dict(state_timeout_s=.1, command_timeout_s=.1, inference_timeout_s=.015,
                       mode_timeout_s=.04, prime_settle_s=.02, subscriber_timeout_s=.02,
                       prepare_timeout_s=.5, ready_timeout_s=1., run_timeout_s=.5,
                       max_trunk_tilt_deg=20., mode_check_interval_s=.005,
                       status_rpc_timeout_s=.02),
        'prepare': dict(start_model_q=list(nominal), start_tolerance_rad=.01,
                        waypoints=[dict(model_q=list(nominal), duration_s=.04)],
                        ready_tolerance_rad=.01, ready_speed_limit_rad_s=.05, ready_dwell_s=.04),
        'stop': {'exit_mode': 'damping'},
    }


class FakeTransport:
    def __init__(self, clock, q):
        self.clock, self.q = clock, tuple(q)
        self.dq = (0.,)*22
        self.commands_sent = 0
        self.enabled = False
        self.inhibited = False
        self.fault = None
        self.mode = 2
        self.sequence = 0
        self.state_time = None
        self.subscriber_error = None
        self.publish_error = None
        self.custom_error = None
        self.exit_error = None
        self.on_subscriber = None
        self.on_custom = None
        self.modes = []
        self.published = []
        self.exit_attempt = threading.Event()

    def latest_state(self):
        self.sequence += 1
        receipt = self.clock() if self.state_time is None else self.state_time
        return StateSnapshot(self.q, self.dq, (0.,)*22, (0.,)*3, (0.,)*3,
                             receipt, self.sequence)

    def enable_motor_interface(self):
        if self.inhibited: raise SafetyFault('inhibited')
        self.enabled = True

    def wait_for_command_subscriber(self, **kwargs):
        if self.subscriber_error: raise SafetyFault(self.subscriber_error)
        if self.on_subscriber: self.on_subscriber()

    def publish(self, q, kp, kd, *, deadline):
        if self.inhibited: raise SafetyFault('inhibited')
        if self.publish_error: raise SafetyFault(self.publish_error)
        if self.clock() >= deadline: raise SafetyFault('expired')
        self.q = tuple(q)
        self.published.append(tuple(q))
        self.commands_sent += 1
        return self.clock()

    def inhibit_commands(self, reason): self.inhibited = True

    def change_mode(self, name, *, abort_event=None, timeout_s=None):
        if name == 'custom':
            if self.on_custom: self.on_custom()
            if self.custom_error: raise SafetyFault(self.custom_error)
            if self.inhibited or (abort_event is not None and abort_event.is_set()):
                raise SafetyFault('custom aborted')
        else:
            self.exit_attempt.set()
            if self.exit_error: raise SafetyFault(self.exit_error)
        self.modes.append(name)
        self.mode = {'custom':3, 'damping':0, 'walking':2}[name]

    def get_mode(self, **kwargs): return self.mode


class RuntimeTests(unittest.TestCase):
    def make_session(self, actor=None, real_clock=False):
        path = BRIDGE.parent/'team15_proprio_package/tasks/team15_proprio/models/proprio_hold.contract.json'
        contract = json.loads(path.read_text(encoding='utf-8'))
        clock = time.monotonic if real_clock else Clock()
        core = ControlCore(session_config(contract), contract)
        transport = FakeTransport(clock, core.nominal)
        events = []
        session = Session(transport, core, actor or (lambda obs: np.zeros(22)), events.append, clock=clock)
        self.addCleanup(lambda: session.request_stop('test cleanup'))
        return session, transport, core, clock, events

    def to_ready(self, session, transport, clock):
        session.request_arm('SIM-ONLY'); session.tick()
        for _ in range(20):
            clock.advance(); session.tick()
            if session.mode_thread is not None: session.mode_thread.join(.2)
            if session.stage == 'READY': return
        self.fail('Session did not reach READY')

    def to_running(self, session, transport, clock):
        self.to_ready(session, transport, clock)
        session.request_run(); clock.advance(); session.tick()
        self.assertEqual(session.stage, 'RUNNING')

    def test_no_motor_or_mode_before_arm_and_serial_must_match(self):
        s,t,c,clock,_ = self.make_session()
        for _ in range(3): clock.advance(); s.tick()
        self.assertFalse(t.enabled); self.assertEqual(t.commands_sent,0); self.assertEqual(t.modes,[])
        with self.assertRaises(SafetyFault): s.request_arm('WRONG-ROBOT')
        with self.assertRaises(SafetyFault): s.request_run()
        self.assertFalse(s.arm_event.is_set())

    def test_prepare_ready_waits_for_manual_run(self):
        s,t,c,clock,_ = self.make_session()
        self.to_ready(s,t,clock)
        for _ in range(5): clock.advance(); s.tick()
        self.assertEqual(s.stage,'READY')
        self.assertEqual(t.modes,['custom'])
        self.assertGreater(t.commands_sent,0)
        s.request_run(); clock.advance(); s.tick()
        self.assertEqual(s.stage,'RUNNING')

    def test_missing_command_subscriber_does_not_prime_or_change_mode(self):
        s,t,c,clock,_ = self.make_session()
        t.subscriber_error='subscriber timeout'; s.request_arm('SIM-ONLY')
        with self.assertRaisesRegex(SafetyFault,'subscriber timeout'): s.tick()
        self.assertEqual(t.commands_sent,0); self.assertEqual(t.modes,[])

    def test_state_rechecked_after_wait_for_subscriber(self):
        s,t,c,clock,_ = self.make_session()
        t.on_subscriber=lambda: setattr(t,'state_time',clock()-.2)
        s.request_arm('SIM-ONLY')
        with self.assertRaisesRegex(SafetyFault,'stale'): s.tick()
        self.assertEqual(t.commands_sent,0); self.assertEqual(t.modes,[])

    def test_ready_requires_low_speed_and_continuous_dwell(self):
        s,t,c,clock,_ = self.make_session()
        t.dq=(.1,)*22
        s.request_arm('SIM-ONLY'); s.tick()
        for _ in range(10):
            clock.advance(); s.tick()
            if s.mode_thread is not None: s.mode_thread.join(.2)
        self.assertEqual(s.stage,'PREPARING')
        t.dq=(0.,)*22
        clock.advance(); s.tick()
        self.assertEqual(s.stage,'PREPARING')
        clock.advance(); s.tick()
        self.assertEqual(s.stage,'PREPARING')
        for _ in range(3): clock.advance(); s.tick()
        self.assertEqual(s.stage,'READY')

    def test_stop_latches_and_session_cannot_rearm(self):
        s,t,c,clock,_ = self.make_session()
        self.to_ready(s,t,clock)
        count=t.commands_sent
        s.request_stop(); s.finish()
        for _ in range(3): clock.advance(); s.tick()
        self.assertEqual(t.commands_sent,count)
        self.assertTrue(s.exit_confirmed)
        self.assertEqual(t.modes,['custom','damping'])
        with self.assertRaises(SafetyFault): s.request_arm('SIM-ONLY')

    def test_stop_during_actor_never_publishes_its_output(self):
        s,t,c,clock,_ = self.make_session()
        self.to_running(s,t,clock)
        def actor(obs):
            s.request_stop('stop during inference')
            return np.ones(22)
        s.actor=actor; count=t.commands_sent
        clock.advance(); s.tick(); s.finish()
        self.assertEqual(t.commands_sent,count)
        self.assertTrue(s.exit_confirmed)

    def test_expired_inference_does_not_commit_previous_action(self):
        s,t,c,clock,_ = self.make_session()
        self.to_running(s,t,clock)
        before=c.previous_action.copy(); count=t.commands_sent
        def actor(obs): clock.advance(.03); return np.ones(22)
        s.actor=actor; clock.advance()
        with self.assertRaisesRegex(SafetyFault,'inference'): s.tick()
        self.assertEqual(t.commands_sent,count)
        np.testing.assert_array_equal(c.previous_action,before)

    def test_publish_failure_does_not_commit_actor_output(self):
        s,t,c,clock,_ = self.make_session(actor=lambda obs:np.ones(22))
        self.to_running(s,t,clock)
        before=c.previous_action.copy(); count=t.commands_sent
        t.publish_error='publisher failed'; clock.advance()
        with self.assertRaisesRegex(SafetyFault,'publisher failed'): s.tick()
        self.assertEqual(t.commands_sent,count)
        np.testing.assert_array_equal(c.previous_action,before)

    def test_stale_original_observation_not_replaced_by_fresh_one(self):
        s,t,c,clock,_ = self.make_session()
        self.to_running(s,t,clock)
        clock.advance(); old=t.latest_state()
        command=c.policy_command(np.zeros(22),old,clock())
        clock.advance(.11)
        self.assertEqual(t.latest_state().received_monotonic,clock())
        count=t.commands_sent
        with self.assertRaisesRegex(SafetyFault,'expired'): s._publish(command)
        self.assertEqual(t.commands_sent,count)

    def test_custom_failure_stops_and_attempts_reviewed_exit(self):
        s,t,c,clock,_ = self.make_session()
        t.custom_error='Custom mode denied'
        s.request_arm('SIM-ONLY'); s.tick(); clock.advance(.03); s.tick()
        s.mode_thread.join(.3); s.finish()
        self.assertTrue(s.stop_event.is_set()); self.assertIn('Custom mode failed',s.reason)
        self.assertTrue(t.exit_attempt.is_set()); self.assertTrue(s.exit_confirmed)

    def test_exit_failure_reported_as_unconfirmed(self):
        s,t,c,clock,events = self.make_session()
        self.to_ready(s,t,clock)
        t.exit_error='RPC failed'; s.request_stop(); s.finish()
        self.assertFalse(s.exit_confirmed)
        self.assertTrue(any(e['event']=='EXIT_UNCONFIRMED' for e in events))

    def test_watchdog_attempts_exit_even_while_inference_is_blocked(self):
        started=threading.Event(); release=threading.Event()
        def blocked_actor(obs):
            started.set(); release.wait(1.); return np.zeros(22)
        s,t,c,clock,_ = self.make_session(actor=blocked_actor,real_clock=True)
        t.enable_motor_interface(); s.motor_enabled=True
        first=c.prime(t.latest_state(),clock()); s._publish(first)
        s._stage('RUNNING'); t.mode=3
        s.config['limits']['command_timeout_s']=.04
        worker=threading.Thread(target=s.run,daemon=True); worker.start()
        self.addCleanup(release.set)
        self.assertTrue(started.wait(.3))
        self.assertTrue(t.exit_attempt.wait(.5),'Exit worker must not await actor return')
        self.assertTrue(worker.is_alive(),'Actor should still be blocked during exit attempt')
        self.assertTrue(t.inhibited)
        release.set(); worker.join(.5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(t.commands_sent,1)

    def test_external_stop_mode_is_not_overridden_by_exit_walking(self):
        s,t,c,clock,_ = self.make_session()
        self.to_ready(s,t,clock)
        s.config['stop']['exit_mode']='walking'
        t.mode=0  # external controller / TA deliberately selected damping
        thread=threading.Thread(target=s.monitor_mode,daemon=True); thread.start()
        thread.join(.3); s.finish()
        self.assertTrue(s.stop_event.is_set())
        self.assertNotIn('walking',t.modes,'Must not override an observed external stop')

    def test_external_mode_observed_while_exit_worker_waits_is_preserved(self):
        s,t,c,clock,_ = self.make_session()
        self.to_ready(s,t,clock)
        s.config['stop']['exit_mode']='walking'
        release=threading.Event()
        s.mode_thread=threading.Thread(target=lambda:release.wait(.3),daemon=True)
        s.mode_thread.start()
        s.request_stop('operator stop')
        self.assertIsNotNone(s.exit_thread)
        # The exit worker exists but waits for the outstanding Custom worker.
        # A concurrent status reply now reports that an external stop won.
        s.external_mode=0
        t.mode=0
        release.set(); s.finish()
        self.assertNotIn('walking',t.modes,'Late observed external mode must still win')


if __name__ == '__main__': unittest.main()
