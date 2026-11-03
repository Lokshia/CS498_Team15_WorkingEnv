"""Read-only collector tests without importing ROS or connecting hardware."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BRIDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE))
from collect_robot_state import collect, main, snapshot_message


def message(count=22):
    return SimpleNamespace(
        imu_state=SimpleNamespace(rpy=[.01, .02, .03], gyro=[.1, .2, .3], acc=[0., 0., 9.81]),
        motor_state_serial=[SimpleNamespace(q=i/100, dq=i/1000, ddq=.5, tau_est=i/10,
                                            mode=1, lost=0, temperature=30, reserve=[0, 42]) for i in range(count)],
        motor_state_parallel=[SimpleNamespace(q=.2, dq=.01, ddq=.2, tau_est=.4,
                                              mode=1, lost=0, temperature=31, reserve=[1, 2]) for _ in range(22)])


class Clock:
    now = 0.
    def __call__(self): return self.now


@contextmanager
def fake_ros(clock, packets=True, serial_count=22, stop_early=False):
    state = SimpleNamespace(subscriptions=[], publisher_attempts=0, client_attempts=0,
                            destroyed=False, context=None, callback=None, spins=0)
    class Context:
        def __init__(self): self.active=True; state.context=self
        def try_shutdown(self): self.active=False
    class Node:
        def create_subscription(self, kind, topic, callback, qos):
            state.subscriptions.append((kind, topic, qos))
            state.callback = callback
        def create_publisher(self, *a, **k):
            state.publisher_attempts += 1
            raise AssertionError('Collector must not create publishers')
        def create_client(self, *a, **k):
            state.client_attempts += 1
            raise AssertionError('Collector must not create clients')
        def destroy_node(self): state.destroyed=True
    rclpy = ModuleType('rclpy')
    rclpy.init=lambda **kw:None
    rclpy.create_node=lambda *a,**kw:Node()
    rclpy.ok=lambda context:context.active
    def spin_once(node, timeout_sec):
        clock.now+=timeout_sec
        state.spins+=1
        if packets: state.callback(message(serial_count))
        if stop_early and state.spins==1: state.context.active=False
    rclpy.spin_once=spin_once
    context=ModuleType('rclpy.context'); context.Context=Context
    qos=ModuleType('rclpy.qos')
    qos.QoSProfile=lambda **kwargs:SimpleNamespace(**kwargs)
    qos.ReliabilityPolicy=SimpleNamespace(BEST_EFFORT='BE')
    qos.HistoryPolicy=SimpleNamespace(KEEP_LAST='KL')
    qos.DurabilityPolicy=SimpleNamespace(VOLATILE='VOL')
    package=ModuleType('booster_interface'); msg=ModuleType('booster_interface.msg')
    msg.LowState=SimpleNamespace
    with patch.dict(sys.modules, {'rclpy':rclpy,'rclpy.context':context,'rclpy.qos':qos,
                                  'booster_interface':package,'booster_interface.msg':msg}):
        yield state


class ProbeTests(unittest.TestCase):
    def test_raw_fields_copied_without_mapping(self):
        original=message()
        snap=snapshot_message(original,10.)
        self.assertEqual(snap['serial_count'],22)
        self.assertEqual(snap['parallel_count'],22)
        self.assertEqual(snap['motor_state_serial'][7],
                         dict(q=.07,dq=.007,ddq=.5,tau_est=.7,mode=1,lost=0,temperature=30,reserve=[0,42]))
        self.assertEqual(snap['imu']['acc'], [0.,0.,9.81])
        self.assertEqual(snap['motor_state_parallel'][0]['reserve'], [1,2])
        original.motor_state_serial[7].reserve[1]=999
        self.assertEqual(snap['motor_state_serial'][7]['reserve'], [0,42])
        original.imu_state.rpy[0]=999.
        self.assertEqual(snap['imu']['rpy'][0],.01)
        self.assertEqual(snap['issues'],[])

    def test_wrong_serial_count_is_retained_and_flagged(self):
        snap=snapshot_message(message(21),1.)
        self.assertEqual(len(snap['motor_state_serial']),21)
        self.assertTrue(any('serial count' in error for error in snap['issues']))

    def test_nonfinite_values_saved_as_null_not_nonstandard_json(self):
        msg=message(); msg.imu_state.gyro[1]=float('nan'); msg.motor_state_serial[0].q=float('inf')
        snap=snapshot_message(msg,1.)
        self.assertIsNone(snap['imu']['gyro'][1]); self.assertIsNone(snap['motor_state_serial'][0]['q'])
        self.assertEqual(len(snap['issues']),2)
        json.dumps(snap,allow_nan=False)

    def test_acc_ddq_and_reserve_validation(self):
        msg=message(); msg.imu_state.acc[0]=float('inf')
        msg.motor_state_serial[0].ddq=float('nan')
        msg.motor_state_serial[1].reserve=[0]
        snap=snapshot_message(msg,1.)
        self.assertIsNone(snap['imu']['acc'][0])
        self.assertIsNone(snap['motor_state_serial'][0]['ddq'])
        self.assertEqual(snap['motor_state_serial'][1]['reserve'],[0])
        self.assertEqual(len(snap['issues']),3)
        json.dumps(snap,allow_nan=False)

    def test_explicit_read_only_qos_and_complete_capture(self):
        clock=Clock()
        with fake_ros(clock) as state:
            report=collect(.2,.5,clock=clock)
        self.assertTrue(report['summary']['passed_read_only_probe'])
        self.assertGreater(report['summary']['callbacks_received'],1)
        self.assertEqual(state.publisher_attempts,0); self.assertEqual(state.client_attempts,0)
        qos=state.subscriptions[0][2]
        self.assertEqual((qos.depth,qos.reliability,qos.durability,qos.history),(1,'BE','VOL','KL'))
        self.assertTrue(state.destroyed); self.assertFalse(state.context.active)

    def test_bounded_initial_timeout_without_messages(self):
        clock=Clock()
        with fake_ros(clock,packets=False):
            report=collect(.2,.3,clock=clock)
        self.assertEqual(report['summary']['stop_reason'],'no_state_before_initial_timeout')
        self.assertEqual(report['summary']['callbacks_received'],0)
        self.assertFalse(report['summary']['passed_read_only_probe'])
        self.assertLessEqual(clock.now,.36)

    def test_wrong_count_still_saves_full_report(self):
        clock=Clock()
        with fake_ros(clock,serial_count=23):
            report=collect(.2,.3,clock=clock)
        self.assertTrue(report['summary']['completed_interval'])
        self.assertEqual(report['summary']['serial_counts_seen'],[23])
        self.assertFalse(report['summary']['serial_count_matches_actor'])
        self.assertFalse(report['summary']['passed_read_only_probe'])
        json.dumps(report,allow_nan=False)

    def test_early_ros_shutdown_is_not_a_complete_test(self):
        clock=Clock()
        with fake_ros(clock,stop_early=True):
            report=collect(.2,.3,clock=clock)
        self.assertEqual(report['summary']['stop_reason'],'ros_context_stopped')
        self.assertFalse(report['summary']['passed_read_only_probe'])

    def test_existing_output_is_never_overwritten_or_connects_ros(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.json'; path.write_text('keep')
            with patch('collect_robot_state.collect',side_effect=AssertionError('should not connect')):
                with self.assertRaises(FileExistsError): main(['--output',str(path)])
            self.assertEqual(path.read_text(),'keep')

    def test_bad_arguments_rejected_before_ros_import(self):
        for seconds in (float('nan'),0.,-1.,61.):
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError): collect(seconds)
        for wait in (0.,11.,float('inf')):
            with self.assertRaises(ValueError): collect(initial_wait_s=wait)


if __name__=='__main__': unittest.main()
