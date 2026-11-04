"""Transport tests with mocked ROS only; no hardware/network is opened."""
from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
import copy
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


BRIDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE))
from team15_hardware.ros_transport import RosTransport, TransportError


def low_state():
    return SimpleNamespace(
        imu_state=SimpleNamespace(rpy=[0., 0., 0.], gyro=[0., 0., 0.]),
        motor_state_serial=[SimpleNamespace(q=i/100, dq=i/1000, tau_est=i/10) for i in range(22)],
    )


class FakeLowCmd:
    CMD_TYPE_SERIAL = 7

    def __init__(self):
        self.cmd_type = None
        self.motor_cmd = []


class FakeContext:
    def __init__(self):
        self.active = True

    def try_shutdown(self):
        self.active = False


class FakeClient:
    def __init__(self):
        self.requests = []
        self.ready = True
        self.pending = False
        self.status = 0
        self.mode = 2
        self.apply_mode = True
        self.bad_body = None
        self.futures = []
        self.on_call = None

    def service_is_ready(self):
        return self.ready

    def wait_for_service(self, timeout_sec):
        time.sleep(timeout_sec)
        return self.ready

    def call_async(self, request):
        self.requests.append(copy.deepcopy(request))
        future = Future()
        self.futures.append(future)
        if self.on_call:
            self.on_call(request)
        if not self.pending:
            if request.msg.api_id == 2000 and self.apply_mode:
                self.mode = json.loads(request.msg.body)["mode"]
            body = self.bad_body if self.bad_body is not None else json.dumps({"current_mode": self.mode})
            future.set_result(SimpleNamespace(msg=SimpleNamespace(status=self.status, body=body)))
        return future


class FakeNode:
    def __init__(self):
        self.subscriptions = []
        self.publishers = []
        self.clients = []
        self.published = []
        self.client = FakeClient()
        self.destroyed = False
        self.command_subscribers = 1
        self.command_publishers = 0

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))

    def create_publisher(self, msg_type, topic, qos):
        pub = SimpleNamespace(
            publish=lambda message: self.published.append(copy.deepcopy(message)),
            get_subscription_count=lambda: self.command_subscribers,
        )
        self.publishers.append((msg_type, topic, qos))
        self.command_publishers += 1
        return pub

    def count_publishers(self, topic):
        return self.command_publishers

    def count_subscribers(self, topic):
        return self.command_subscribers

    def create_client(self, service_type, name):
        self.clients.append((service_type, name))
        return self.client

    def destroy_node(self):
        self.destroyed = True


class FakeExecutor:
    def __init__(self, context):
        self.context = context
        self.node = None

    def add_node(self, node):
        self.node = node

    def spin_once(self, timeout_sec):
        time.sleep(min(timeout_sec, .001))

    def shutdown(self, timeout_sec):
        pass


@contextmanager
def mock_ros():
    node = FakeNode()
    names = ["rclpy", "rclpy.context", "rclpy.executors", "rclpy.qos",
             "booster_interface", "booster_interface.msg", "booster_interface.srv"]
    modules = {name: ModuleType(name) for name in names}
    ros = modules["rclpy"]
    ros.init = lambda **kw: None
    ros.ok = lambda context: context.active
    ros.create_node = lambda *a, **kw: node
    modules["rclpy.context"].Context = FakeContext
    modules["rclpy.executors"].SingleThreadedExecutor = FakeExecutor
    qos = modules["rclpy.qos"]
    qos.QoSProfile = lambda **kw: SimpleNamespace(**kw)
    qos.ReliabilityPolicy = SimpleNamespace(BEST_EFFORT="best_effort", RELIABLE="reliable")
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")
    msg = modules["booster_interface.msg"]
    msg.LowState = type("LowState", (), {})
    msg.LowCmd = FakeLowCmd
    msg.MotorCmd = SimpleNamespace
    msg.BoosterApiReqMsg = SimpleNamespace
    modules["booster_interface.srv"].RpcService = SimpleNamespace(Request=SimpleNamespace)
    with patch.dict(sys.modules, modules):
        yield node


class TransportTests(unittest.TestCase):
    def start(self, *, motor=False, settings=None, clock=time.monotonic):
        transport = RosTransport(settings or {"motor_weight": [0.] * 22, "rpc_timeout_s": .06}, clock=clock)
        transport.start()
        self.addCleanup(transport.close)
        if motor:
            transport.enable_motor_interface()
        return transport

    def test_import_does_not_require_ros(self):
        path = BRIDGE / "team15_hardware/ros_transport.py"
        spec = importlib.util.spec_from_file_location("transport_without_ros", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"rclpy": None, "booster_interface": None,
                                     "transport_without_ros": module}):
            spec.loader.exec_module(module)
            self.assertIsNotNone(module.RosTransport({}))

    def test_read_only_no_publisher_no_client_or_mode(self):
        with mock_ros() as node:
            transport = self.start()
            self.assertEqual(len(node.subscriptions), 1)
            self.assertEqual(node.publishers, [])
            self.assertEqual(node.clients, [])
            self.assertEqual(node.published, [])
            self.assertEqual(node.subscriptions[0][3].reliability, "best_effort")
            self.assertEqual(node.subscriptions[0][3].depth, 1)
            with self.assertRaises(TransportError):
                transport.publish([0]*22, [1]*22, [1]*22, deadline=time.monotonic()+1)
            with self.assertRaises(TransportError):
                transport.change_mode("custom")
            self.assertEqual(transport.commands_sent, 0)

    def test_snapshot_copies_one_message_and_is_immutable(self):
        with mock_ros():
            transport = self.start(clock=lambda: 42.)
            message = low_state()
            transport._on_state(message)
            snapshot = transport.latest_state()
            self.assertEqual(snapshot.sequence, 1)
            self.assertEqual(snapshot.received_monotonic, 42.)
            self.assertEqual(snapshot.q, tuple(i/100 for i in range(22)))
            message.motor_state_serial[0].q = 99
            self.assertEqual(snapshot.q[0], 0.)
            with self.assertRaises(FrozenInstanceError):
                snapshot.sequence = 99
            with self.assertRaises(TypeError):
                snapshot.q[0] = 99

    def test_malformed_state_latches_fault_and_no_recovery(self):
        with mock_ros():
            transport = self.start(motor=True)
            transport._on_state(low_state())
            before = transport.latest_state()
            message = low_state()
            message.motor_state_serial.pop()
            transport._on_state(message)
            self.assertIn("22", transport.fault)
            transport._on_state(low_state())
            self.assertIs(transport.latest_state(), before)
            with self.assertRaises(TransportError):
                transport.publish([0]*22, [1]*22, [1]*22, deadline=time.monotonic()+1)

    def test_all_sensor_fields_require_finite_numbers(self):
        for field in ("q", "dq", "tau_est", "rpy", "gyro"):
            with self.subTest(field=field), mock_ros():
                transport = self.start()
                message = low_state()
                if field in ("rpy", "gyro"):
                    getattr(message.imu_state, field)[0] = float("nan")
                else:
                    setattr(message.motor_state_serial[0], field, float("inf"))
                transport._on_state(message)
                self.assertIsNotNone(transport.fault)
                self.assertIsNone(transport.latest_state())
                transport.close()

    def test_motor_weight_must_be_explicit(self):
        with mock_ros() as node:
            transport = self.start(settings={"rpc_timeout_s": .06})
            with self.assertRaisesRegex(TransportError, "motor_weight"):
                transport.enable_motor_interface()
            self.assertEqual(node.publishers, [])
            self.assertEqual(node.clients, [])

    def test_enable_never_publishes_or_changes_mode(self):
        with mock_ros() as node:
            self.start(motor=True)
            self.assertEqual(len(node.publishers), 1)
            self.assertEqual(len(node.clients), 1)
            self.assertEqual(node.publishers[0][2].reliability, "reliable")
            self.assertEqual(node.published, [])
            self.assertEqual(node.client.requests, [])

    def test_serial_command_fields_and_counter(self):
        with mock_ros() as node:
            weights = [i/22 for i in range(22)]
            transport = self.start(motor=True, settings={"motor_weight": weights})
            q = [i*.01 for i in range(22)]
            transport.publish(q, [2]*22, [.5]*22, deadline=time.monotonic()+1)
            message = node.published[0]
            self.assertEqual(message.cmd_type, FakeLowCmd.CMD_TYPE_SERIAL)
            self.assertEqual(len(message.motor_cmd), 22)
            self.assertEqual([m.q for m in message.motor_cmd], q)
            self.assertEqual([m.weight for m in message.motor_cmd], weights)
            self.assertTrue(all(m.dq == 0 and m.tau == 0 and m.kp == 2 and m.kd == .5 for m in message.motor_cmd))
            self.assertEqual(transport.commands_sent, 1)

    def test_subscriber_wait_and_lost_subscriber_are_bounded(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            transport.wait_for_command_subscriber(timeout_s=.03)
            node.command_subscribers = 0
            start = time.monotonic()
            with self.assertRaisesRegex(TransportError, "timed out"):
                transport.wait_for_command_subscriber(timeout_s=.02)
            self.assertLess(time.monotonic()-start,.3)
            with self.assertRaisesRegex(TransportError, "No ROS subscriber"):
                transport.publish([0]*22,[1]*22,[1]*22,deadline=time.monotonic()+1)
            self.assertEqual(node.published,[])

    def test_competing_command_publisher_latches_fault(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            self.assertEqual(transport.command_graph_counts(), {"publishers":1,"subscribers":1})
            node.command_publishers=2
            with self.assertRaisesRegex(TransportError,"Competing"):
                transport.wait_for_command_subscriber(timeout_s=.03)
            self.assertIsNotNone(transport.fault)
            node.command_publishers=1
            with self.assertRaises(TransportError):
                transport.publish([0]*22,[1]*22,[1]*22,deadline=time.monotonic()+1)
            self.assertEqual(node.published,[])

    def test_own_publisher_discovery_must_complete_before_prime(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            node.command_publishers=0
            with self.assertRaisesRegex(TransportError,"timed out"):
                transport.wait_for_command_subscriber(timeout_s=.02)
            self.assertEqual(node.published,[])

    def test_expired_or_nonfinite_or_wrong_length_commands_rejected(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            for q,kp,kd,deadline in (([0]*21,[1]*22,[1]*22,time.monotonic()+1),
                                     ([float("nan")]*22,[1]*22,[1]*22,time.monotonic()+1),
                                     ([0]*22,[-1]*22,[1]*22,time.monotonic()+1),
                                     ([0]*22,[1]*22,[1]*22,time.monotonic()-1),
                                     ([0]*22,[1]*22,[1]*22,float("inf"))):
                with self.assertRaises(TransportError):
                    transport.publish(q,kp,kd,deadline=deadline)
            self.assertEqual(node.published, [])

    def test_stop_latch_blocks_publication_and_custom_but_allows_exit(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            transport.inhibit_commands("operator stop")
            with self.assertRaises(TransportError):
                transport.publish([0]*22,[1]*22,[1]*22,deadline=time.monotonic()+1)
            with self.assertRaises(TransportError):
                transport.change_mode("custom")
            transport.change_mode("damping")
            self.assertEqual(node.client.mode, 0)
            self.assertEqual(node.published, [])

    def test_rpc_verifies_mode_and_does_not_repeat_change(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            transport.change_mode("custom")
            self.assertEqual([r.msg.api_id for r in node.client.requests], [2000,2018])
            self.assertEqual(transport.get_mode(), 3)
            self.assertEqual(node.published, [])

    def test_wrong_mode_timeout_is_bounded_and_reported(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            node.client.apply_mode = False
            start = time.monotonic()
            with self.assertRaisesRegex(TransportError, "timed out"):
                transport.change_mode("damping",timeout_s=.03)
            self.assertLess(time.monotonic()-start,.3)
            self.assertEqual(sum(r.msg.api_id==2000 for r in node.client.requests),1)

    def test_unavailable_service_times_out_without_request(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            node.client.ready = False
            start = time.monotonic()
            with self.assertRaisesRegex(TransportError,"timed out"):
                transport.change_mode("custom",timeout_s=.03)
            self.assertLess(time.monotonic()-start,.3)
            self.assertEqual(node.client.requests,[])

    def test_pending_rpc_cancel_and_abort_are_bounded(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            node.client.pending = True
            with self.assertRaisesRegex(TransportError,"timed out"):
                transport.change_mode("custom",timeout_s=.02)
            self.assertTrue(node.client.futures[-1].cancelled())
            abort = threading.Event(); abort.set()
            count = len(node.client.requests)
            with self.assertRaisesRegex(TransportError,"aborted"):
                transport.change_mode("custom",abort_event=abort)
            self.assertEqual(len(node.client.requests),count)

    def test_stop_during_custom_rpc_never_reports_custom_success(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            abort=threading.Event()
            def stop_on_change(req):
                if req.msg.api_id==2000:
                    transport.inhibit_commands("stop during mode request")
                    abort.set()
            node.client.on_call=stop_on_change
            with self.assertRaisesRegex(TransportError,"aborted"):
                transport.change_mode("custom",abort_event=abort)
            node.client.on_call=None
            transport.change_mode("damping")
            self.assertEqual(node.client.mode,0)
            self.assertEqual(node.published,[])

    def test_stop_while_waiting_for_service_prevents_custom_request(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            node.client.ready = False
            def wait_then_stop(timeout_sec):
                transport.inhibit_commands("stop during service wait")
                node.client.ready = True
            node.client.wait_for_service = wait_then_stop
            with self.assertRaisesRegex(TransportError,"inhibited"):
                transport.change_mode("custom")
            self.assertEqual(node.client.requests,[])

    def test_exit_rpc_serializes_after_cancelled_custom_request(self):
        with mock_ros() as node:
            transport=self.start(motor=True)
            node.client.pending=True
            abort=threading.Event()
            outcomes={}
            def enter():
                try:
                    transport.change_mode('custom',abort_event=abort,timeout_s=.25)
                    outcomes['custom']='returned'
                except Exception as exc:
                    outcomes['custom']=str(exc)
            def exit_mode():
                try:
                    transport.change_mode('damping',timeout_s=.25)
                    outcomes['exit']='returned'
                except Exception as exc:
                    outcomes['exit']=str(exc)
            first=threading.Thread(target=enter)
            first.start()
            deadline=time.monotonic()+.2
            while not node.client.requests and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertEqual(len(node.client.requests),1)
            second=threading.Thread(target=exit_mode)
            second.start()
            time.sleep(.01)
            self.assertEqual(len(node.client.requests),1)
            transport.inhibit_commands('operator stop')
            node.client.pending=False
            abort.set()
            first.join(.3); second.join(.3)
            self.assertFalse(first.is_alive()); self.assertFalse(second.is_alive())
            self.assertIn('aborted',outcomes['custom'])
            self.assertEqual(outcomes['exit'],'returned')
            changes=[json.loads(r.msg.body)['mode'] for r in node.client.requests if r.msg.api_id==2000]
            self.assertEqual(changes,[3,0])
            self.assertEqual(node.client.mode,0)
            self.assertEqual(node.published,[])

    def test_exit_rpc_failure_is_not_hidden(self):
        with mock_ros() as node:
            transport = self.start(motor=True)
            transport._latch_fault("bad state")
            transport.inhibit_commands("stop")
            node.client.status=7
            with self.assertRaisesRegex(TransportError,"failed"):
                transport.change_mode("damping")

    def test_status_must_be_well_formed(self):
        for body in ('not-json','[]','{}','{"current_mode":true}','{"current_mode":"3"}'):
            with self.subTest(body=body),mock_ros() as node:
                transport=self.start(motor=True)
                node.client.bad_body=body
                with self.assertRaises(TransportError): transport.get_mode()
                transport.close()

    def test_close_is_idempotent_and_does_not_switch_mode(self):
        with mock_ros() as node:
            transport=self.start(motor=True)
            transport.close(); transport.close()
            self.assertEqual(node.client.requests,[])
            self.assertTrue(node.destroyed)
            with self.assertRaises(TransportError): transport.start()


if __name__ == "__main__":
    unittest.main()
