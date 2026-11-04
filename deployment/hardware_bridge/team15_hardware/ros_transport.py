"""Bounded ROS 2 I/O for the reviewed K1 serial-joint interface.

Importing this module does not import ROS or create any network interface.
``start()`` is read-only. Publishers and mode RPC clients are created only by
an explicit ``enable_motor_interface()`` call after the caller's preflight.
Publication means accepted by the local ROS publisher, NOT acknowledged by a
motor. Firmware watchdogs and motor-command semantics must be verified on site.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Callable


JOINT_COUNT = 22
MODE_VALUES = {"damping": 0, "walking": 2, "custom": 3}
CHANGE_MODE_API = 2000
GET_STATUS_API = 2018


class TransportError(RuntimeError):
    """An I/O, message, freshness, cancellation or mode-verification failure."""


def _vector(values, count: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransportError(f"{name}: expected {count} finite numbers") from exc
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise TransportError(f"{name}: expected {count} finite numbers")
    return result


@dataclass(frozen=True)
class StateSnapshot:
    """Raw serial-order SI-unit fields, copied from one complete LowState.

    Receipt time measures local freshness only; it does not validate a sensor
    source timestamp. Coordinate/sign/zero transformations belong to the core.
    """

    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau: tuple[float, ...]
    rpy: tuple[float, ...]
    gyro: tuple[float, ...]
    received_monotonic: float
    sequence: int


class RosTransport:
    def __init__(self, settings: dict, clock: Callable[[], float] = time.monotonic):
        self.settings = dict(settings)
        self.clock = clock
        self._lock = threading.RLock()
        self._rpc_lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot = None
        self._fault = None
        self._inhibit_reason = None
        self._sequence = 0
        self.commands_sent = 0
        self._started = False
        self._closed = False
        self._motor_enabled = False
        self._publisher = self._client = None
        self._node = self._executor = self._thread = self._context = None
        self._rclpy = None
        self._weights = None
        self._rpc_timeout = float(self.settings.get("rpc_timeout_s", 2.0))
        if not math.isfinite(self._rpc_timeout) or self._rpc_timeout <= 0:
            raise ValueError("rpc_timeout_s must be finite and positive")
        for key, default in (("low_state_topic", "/low_state"),
                             ("joint_ctrl_topic", "/joint_ctrl"),
                             ("rpc_service", "/booster_rpc_service")):
            value = self.settings.get(key, default)
            if not isinstance(value, str) or not value.startswith("/") or any(c.isspace() for c in value):
                raise ValueError(f"{key} must be an absolute ROS name")
            self.settings[key] = value

    @property
    def fault(self) -> str | None:
        with self._lock:
            return self._fault

    @property
    def motor_enabled(self) -> bool:
        with self._lock:
            return self._motor_enabled

    def _latch_fault(self, reason: str) -> None:
        with self._lock:
            if self._fault is None:
                self._fault = str(reason)

    def start(self) -> None:
        """Start a dedicated-context subscriber; never create command handles."""
        if self._started or self._closed:
            raise TransportError("Transport cannot be restarted")
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from booster_interface.msg import LowState

        self._rclpy = rclpy
        self._context = Context()
        try:
            rclpy.init(args=None, context=self._context)
            self._node = rclpy.create_node("team15_hardware_bridge", context=self._context)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
            self._node.create_subscription(LowState, self.settings["low_state_topic"],
                                           self._on_state, qos)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._started = True
            self._thread = threading.Thread(target=self._spin, name="team15-ros-state", daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    def _spin(self) -> None:
        try:
            while not self._stop.is_set() and self._rclpy.ok(context=self._context):
                self._executor.spin_once(timeout_sec=0.02)
            if not self._stop.is_set():
                self._latch_fault("ROS context stopped unexpectedly")
        except Exception as exc:
            if not self._stop.is_set():
                self._latch_fault(f"ROS executor failed: {type(exc).__name__}: {exc}")

    def _on_state(self, message) -> None:
        try:
            motors = tuple(message.motor_state_serial)
            if len(motors) != JOINT_COUNT:
                raise TransportError(f"Expected exactly {JOINT_COUNT} serial motor states")
            fields = {
                "q": _vector((m.q for m in motors), JOINT_COUNT, "motor q"),
                "dq": _vector((m.dq for m in motors), JOINT_COUNT, "motor dq"),
                "tau": _vector((m.tau_est for m in motors), JOINT_COUNT, "motor tau_est"),
                "rpy": _vector(message.imu_state.rpy, 3, "IMU rpy"),
                "gyro": _vector(message.imu_state.gyro, 3, "IMU gyro"),
            }
            receipt = float(self.clock())
            if not math.isfinite(receipt):
                raise TransportError("Non-finite local receipt time")
            with self._lock:
                if self._fault is not None or self._closed:
                    return
                self._sequence += 1
                self._snapshot = StateSnapshot(**fields, received_monotonic=receipt,
                                               sequence=self._sequence)
        except Exception as exc:
            self._latch_fault(f"Invalid LowState: {type(exc).__name__}: {exc}")

    def latest_state(self) -> StateSnapshot | None:
        with self._lock:
            return self._snapshot

    def enable_motor_interface(self) -> None:
        """Explicit live-only step. Does not publish or request a robot mode."""
        with self._lock:
            self._require_writable()
            if self._motor_enabled:
                raise TransportError("Motor interface already enabled")
            if "motor_weight" not in self.settings:
                raise TransportError("TA must explicitly supply all 22 motor_weight values")
            weights = _vector(self.settings["motor_weight"], JOINT_COUNT, "motor_weight")
            if any(w < 0 or w > 1 for w in weights):
                raise TransportError("motor_weight must be in [0,1]; semantics require on-site review")
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from booster_interface.msg import LowCmd, MotorCmd, BoosterApiReqMsg
            from booster_interface.srv import RpcService
            self._LowCmd, self._MotorCmd = LowCmd, MotorCmd
            self._ApiReq, self._RpcService = BoosterApiReqMsg, RpcService
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
            try:
                self._publisher = self._node.create_publisher(LowCmd, self.settings["joint_ctrl_topic"], qos)
                self._client = self._node.create_client(RpcService, self.settings["rpc_service"])
            except Exception as exc:
                self._latch_fault(f"Motor interface creation failed: {exc}")
                raise TransportError(self._fault) from exc
            self._weights = weights
            self._motor_enabled = True

    def _require_writable(self) -> None:
        if not self._started or self._closed or self._stop.is_set():
            raise TransportError("Transport is not running")
        if self._fault is not None:
            raise TransportError(self._fault)
        if self._inhibit_reason is not None:
            raise TransportError(f"Commands inhibited: {self._inhibit_reason}")

    def inhibit_commands(self, reason: str) -> None:
        """Latch a stop. Shares publish lock, so subsequent publish cannot pass."""
        with self._lock:
            if self._inhibit_reason is None:
                self._inhibit_reason = str(reason)

    def command_graph_counts(self) -> dict[str, int]:
        """Observed ROS graph counts, not authenticated ownership or motor ACK."""
        if not self._started or self._closed:
            raise TransportError("Transport is not running")
        topic = self.settings["joint_ctrl_topic"]
        return {"publishers": int(self._node.count_publishers(topic)),
                "subscribers": int(self._node.count_subscribers(topic))}

    def wait_for_command_subscriber(self, *, abort_event=None, timeout_s=None) -> None:
        """Bounded wait for a ROS command subscriber before priming Custom.

        Discovery alone does not authenticate the subscriber or confirm motor
        delivery. Its identity and command ownership still need site review.
        """
        deadline = self.clock() + self._timeout(timeout_s)
        while True:
            self._check_rpc_wait(deadline, abort_event)
            with self._lock:
                self._require_writable()
                if not self._motor_enabled:
                    raise TransportError("Read-only transport has no command publisher")
                counts = self.command_graph_counts()
                if counts["publishers"] > 1:
                    self._latch_fault("Competing ROS motor-command publisher detected")
                    raise TransportError(self._fault)
                if counts["publishers"] == 1 and self._publisher.get_subscription_count() > 0:
                    return
            self._stop.wait(min(0.01, max(0.0, deadline-self.clock())))

    def publish(self, q, kp, kd, *, deadline: float) -> float:
        """Publish one complete serial position command before its deadline.

        q/kp/kd are already mapped and checked by the core; this repeats format
        and nonnegative-gain checks. No torque feed-forward or velocity target
        is requested. Successful return is local publication, not motor ACK.
        """
        q = _vector(q, JOINT_COUNT, "command q")
        kp = _vector(kp, JOINT_COUNT, "command kp")
        kd = _vector(kd, JOINT_COUNT, "command kd")
        if any(v < 0 for v in kp + kd):
            raise TransportError("Negative motor gain")
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise TransportError("Command deadline must be finite")
        with self._lock:
            self._require_writable()
            if not self._motor_enabled:
                raise TransportError("Read-only transport cannot publish")
            if self.command_graph_counts()["publishers"] != 1:
                self._latch_fault("Expected exactly this motor-command publisher in ROS graph")
                raise TransportError(self._fault)
            if self._publisher.get_subscription_count() < 1:
                raise TransportError("No ROS subscriber for motor commands")
            if self.clock() >= deadline:
                raise TransportError("Command expired before publication")
            message = self._LowCmd()
            message.cmd_type = self._LowCmd.CMD_TYPE_SERIAL
            motors = []
            for i in range(JOINT_COUNT):
                motor = self._MotorCmd()
                motor.q, motor.dq, motor.tau = q[i], 0.0, 0.0
                motor.kp, motor.kd, motor.weight = kp[i], kd[i], self._weights[i]
                motors.append(motor)
            message.motor_cmd.extend(motors)
            if self.clock() >= deadline:
                raise TransportError("Command expired while building publication")
            try:
                self._publisher.publish(message)
            except Exception as exc:
                self._latch_fault(f"Command publish failed: {exc}")
                raise TransportError(self._fault) from exc
            self.commands_sent += 1
            return float(self.clock())

    def _timeout(self, timeout_s) -> float:
        value = self._rpc_timeout if timeout_s is None else float(timeout_s)
        if not math.isfinite(value) or value <= 0:
            raise TransportError("RPC timeout must be finite and positive")
        return value

    def _check_rpc_wait(self, deadline, abort_event) -> None:
        if self._stop.is_set() or self._closed:
            raise TransportError("Transport closed during RPC")
        if abort_event is not None and abort_event.is_set():
            raise TransportError("Mode RPC aborted")
        if self.clock() >= deadline:
            raise TransportError("Mode RPC timed out")
        if not self._rclpy.ok(context=self._context):
            raise TransportError("ROS stopped during RPC")

    def _rpc(self, api_id, body, deadline, abort_event):
        self._check_rpc_wait(deadline, abort_event)
        while not self._client.service_is_ready():
            self._check_rpc_wait(deadline, abort_event)
            self._client.wait_for_service(timeout_sec=min(0.02, max(0.0, deadline-self.clock())))
        self._check_rpc_wait(deadline, abort_event)
        request = self._RpcService.Request()
        request.msg = self._ApiReq()
        request.msg.api_id = api_id
        request.msg.body = "" if body is None else json.dumps(body)
        if api_id == CHANGE_MODE_API and body == {"mode": MODE_VALUES["custom"]}:
            # Atomically reject a late Custom request after STOP. The response
            # wait stays outside this lock so inhibit remains prompt.
            with self._lock:
                self._require_writable()
                self._check_rpc_wait(deadline, abort_event)
                future = self._client.call_async(request)
        else:
            future = self._client.call_async(request)
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        try:
            while not future.done():
                self._check_rpc_wait(deadline, abort_event)
                ready.wait(min(0.01, max(0.0, deadline-self.clock())))
            self._check_rpc_wait(deadline, abort_event)
            result = future.result()
        except Exception:
            future.cancel()
            raise
        if result is None or result.msg.status != 0:
            raise TransportError(f"Booster RPC {api_id} failed")
        try:
            parsed = json.loads(result.msg.body) if result.msg.body else None
        except (ValueError, TypeError) as exc:
            raise TransportError("Booster RPC returned malformed JSON") from exc
        if parsed is not None and not isinstance(parsed, dict):
            raise TransportError("Booster RPC body must be a JSON object")
        return parsed

    def _acquire_rpc(self, timeout_s, abort_event):
        deadline = self.clock() + self._timeout(timeout_s)
        if not self._motor_enabled or self._client is None:
            raise TransportError("Read-only transport cannot issue mode RPCs")
        while True:
            self._check_rpc_wait(deadline, abort_event)
            if self._rpc_lock.acquire(timeout=min(0.01, max(0.0, deadline-self.clock()))):
                return deadline

    @staticmethod
    def _mode_from_status(status):
        value = status.get("current_mode") if isinstance(status, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise TransportError("GET_STATUS omitted a valid current_mode")
        return int(value)

    def get_mode(self, *, abort_event=None, timeout_s=None) -> int:
        """Read current mode via the live interface; no mode-change request."""
        deadline = self._acquire_rpc(timeout_s, abort_event)
        try:
            return self._mode_from_status(self._rpc(GET_STATUS_API, None, deadline, abort_event))
        finally:
            self._rpc_lock.release()

    def change_mode(self, name: str, *, abort_event=None, timeout_s=None) -> None:
        """One bounded CHANGE request, then bounded GET_STATUS verification.

        Exit mode requests remain possible after a latched input/publication
        fault or command inhibit. Entering custom is forbidden in that state.
        Timeout/cancellation does not prove the remote robot ignored a request;
        callers must handle an uncertain mode and use the laboratory stop path.
        """
        if name not in MODE_VALUES:
            raise TransportError(f"Unsupported mode: {name}")
        if name == "custom":
            with self._lock:
                self._require_writable()
        deadline = self._acquire_rpc(timeout_s, abort_event)
        try:
            if name == "custom":
                with self._lock:
                    self._require_writable()
            self._rpc(CHANGE_MODE_API, {"mode": MODE_VALUES[name]}, deadline, abort_event)
            while True:
                status = self._rpc(GET_STATUS_API, None, deadline, abort_event)
                if self._mode_from_status(status) == MODE_VALUES[name]:
                    if name == "custom":
                        with self._lock:
                            self._require_writable()
                    return
                self._check_rpc_wait(deadline, abort_event)
                self._stop.wait(min(0.02, max(0.0, deadline-self.clock())))
        finally:
            self._rpc_lock.release()

    def close(self) -> None:
        """Close local I/O only; never silently select an exit robot mode."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            if self._inhibit_reason is None:
                self._inhibit_reason = "transport closed"
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=0.2)
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            self._context.try_shutdown()

