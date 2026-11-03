"""Robot-independent calibrated observation and bounded position-command path.

No box state, ROS, mode switch, or publisher is used here. Feedback torque
checks are fault detection, NOT motor torque limiting. Firmware protections
and site calibration are required independently.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

N = 22
DT = 0.02
MODEL_SHA = 'c75c00f943ddca0a94ed6722a9848a2443909150aaffc63d253f184b4b55b921'
CONTRACT_SHA = 'c22b6e5138c5ce085afebab39304cf4649e02d1bd39b78bd15c0f9a50107e2d7'
# Additional policy envelope from the frozen deployment model_audit. These
# are NOT hardware limits; live targets also obey narrower reviewed site limits.
VALIDATED_POLICY_RANGES = (
    (-1.012, 1.012), (-.314, .794), (-2.932, 1.196), (-1.642, 1.629),
    (-1.885, 1.885), (-2.242, .816), (-2.932, 1.196), (-1.642, 1.629),
    (-1.885, 1.885), (-.816, 2.242), (-2.958, 2.226), (-.375, 1.536),
    (-1.012, 1.012), (0., 2.321), (-.87, .345), (-.345, .345),
    (-2.958, 2.226), (-1.536, .375), (-1.012, 1.012), (0., 2.321),
    (-.87, .345), (-.345, .345),
)


class SafetyFault(RuntimeError):
    pass


def vector(value, n, label):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise SafetyFault(f'{label}: expected {n} finite numbers') from exc
    if result.shape != (n,) or not np.isfinite(result).all():
        raise SafetyFault(f'{label}: expected {n} finite numbers')
    return result


def world_from_imu(rpy):
    """Extrinsic roll/pitch/yaw radians: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = vector(rpy, 3, 'imu.rpy')
    cr, cp, cy = np.cos([roll, pitch, yaw]); sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


@dataclass(frozen=True)
class Command:
    model_q: np.ndarray
    serial_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effective_action: np.ndarray
    observation_sequence: int
    observation_received: float
    deadline: float


class Actor:
    def __init__(self, path: Path):
        import onnxruntime as ort
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA:
            raise SafetyFault('ONNX hash differs from the validated actor')
        contract_path = path.with_name('proprio_hold.contract.json')
        if hashlib.sha256(contract_path.read_bytes()).hexdigest() != CONTRACT_SHA:
            raise SafetyFault('Actor contract bytes differ from the reviewed contract')
        self.contract = json.loads(contract_path.read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != 'obs' or inputs[0].shape != [1, 72]
                or inputs[0].type != 'tensor(float)' or len(outputs) != 1
                or outputs[0].name != 'actions' or outputs[0].shape != [1, 22]):
            raise SafetyFault('Unexpected ONNX input/output contract')

    def __call__(self, obs):
        obs = vector(obs, 72, 'actor observation').astype(np.float32)
        result = self.session.run(['actions'], {'obs': obs[None]})[0]
        if result.shape != (1, N):
            raise SafetyFault('Actor output shape differs')
        return vector(result[0], N, 'actor output').clip(-1., 1.)

    def verify_vectors(self, path):
        with np.load(path, allow_pickle=False) as data:
            # Export keys are inspected explicitly; never accept a missing oracle.
            obs, expected = data['observations'], data['expected_actions']
            actual = np.stack([self(row) for row in obs])
            maximum = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, atol=2e-5, rtol=1e-5):
                raise SafetyFault(f'Actor test-vector mismatch: {maximum}')
            return {'test_vectors': len(obs), 'max_abs_error': maximum}


class ControlCore:
    def __init__(self, config: dict, contract: dict):
        self.config = config
        mapping = contract['joint_mapping']
        self.nominal = vector(mapping['default_joint_pos_rad'], N, 'default pose')
        self.scale = vector(mapping['action_scale_rad'], N, 'action scale')
        rows = config['joints']
        self.index = np.array([r['serial_index'] for r in rows], dtype=int)
        self.sign = np.array([r['sign'] for r in rows], dtype=float)
        self.offset = np.array([r['zero_offset_rad'] for r in rows], dtype=float)
        self.lower = np.array([r['model_min_rad'] for r in rows])
        self.upper = np.array([r['model_max_rad'] for r in rows])
        self.policy_lower = np.maximum(self.lower, np.array(VALIDATED_POLICY_RANGES)[:, 0])
        self.policy_upper = np.minimum(self.upper, np.array(VALIDATED_POLICY_RANGES)[:, 1])
        self.rate = np.array([r['target_rate_limit_rad_s'] for r in rows])
        self.max_speed = np.array([r['measured_speed_limit_rad_s'] for r in rows])
        self.max_torque = np.array([r['measured_torque_limit_nm'] for r in rows])
        self.max_error = np.array([r['tracking_error_limit_rad'] for r in rows])
        self.kp = self.serial_values([r['kp'] for r in rows])
        self.kd = self.serial_values([r['kd'] for r in rows])
        self.rotation = np.asarray(config['imu']['body_from_imu'], dtype=float)
        self.limits = config['limits']
        self.previous_target = None
        self.previous_action = np.zeros(N)
        self.last_sent_time = None

    def serial_values(self, canonical_values):
        result = np.empty(N)
        result[self.index] = vector(canonical_values, N, 'joint values')
        return result

    def to_serial(self, canonical_q):
        return self.serial_values(self.offset + self.sign*canonical_q)

    def state(self, snapshot, now):
        if snapshot is None:
            raise SafetyFault('No robot state received')
        age = now - snapshot.received_monotonic
        if not np.isfinite(age) or age < -1e-6 or age > self.limits['state_timeout_s']:
            raise SafetyFault(f'Robot state stale/invalid age: {age}')
        q = self.sign*(vector(snapshot.q, N, 'serial q')[self.index] - self.offset)
        dq = self.sign*vector(snapshot.dq, N, 'serial dq')[self.index]
        tau = self.sign*vector(snapshot.tau, N, 'serial torque')[self.index]
        gravity = self.rotation @ world_from_imu(snapshot.rpy).T @ np.array([0., 0., -1.])
        gyro = self.rotation @ vector(snapshot.gyro, 3, 'imu gyro')
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise SafetyFault('Measured joint position outside reviewed site limits')
        if np.any(np.abs(dq) > self.max_speed):
            raise SafetyFault('Measured joint speed exceeds reviewed limit')
        if np.any(np.abs(tau) > self.max_torque):
            raise SafetyFault('Measured joint torque exceeds reviewed limit')
        if -gravity[2] < np.cos(np.deg2rad(self.limits['max_trunk_tilt_deg'])):
            raise SafetyFault('Trunk tilt exceeds reviewed limit')
        return q, dq, gravity, gyro

    def observation(self, snapshot, now):
        q, dq, gravity, gyro = self.state(snapshot, now)
        return np.concatenate([gyro, gravity, q-self.nominal, dq, self.previous_action]).astype(np.float32)

    def prime(self, snapshot, now):
        q, *_ = self.state(snapshot, now)
        start = vector(self.config['prepare']['start_model_q'], N, 'start pose')
        if np.max(np.abs(q-start)) > self.config['prepare']['start_tolerance_rad']:
            raise SafetyFault('Robot is not in the TA-reviewed starting pose')
        return self._command(q, snapshot, now)

    def _command(self, target, snapshot, now):
        target = vector(target, N, 'target')
        q, *_ = self.state(snapshot, now)
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise SafetyFault('Command outside reviewed joint range')
        if np.any(np.abs(target-q) > self.max_error):
            raise SafetyFault('Command tracking error exceeds reviewed limit')
        age_deadline = snapshot.received_monotonic + self.limits['state_timeout_s']
        deadline = min(age_deadline, now + self.limits['command_timeout_s'])
        effective = (target-self.nominal)/self.scale
        return Command(target.copy(), self.to_serial(target), self.kp.copy(), self.kd.copy(),
                       effective, snapshot.sequence, snapshot.received_monotonic, deadline)

    def target_command(self, desired, snapshot, now):
        if self.previous_target is None or self.last_sent_time is None:
            raise SafetyFault('No successfully published starting target')
        elapsed = now-self.last_sent_time
        if elapsed < 0 or elapsed > self.limits['command_timeout_s']:
            raise SafetyFault('Command stream missed its reviewed deadline')
        desired = vector(desired, N, 'desired target').clip(self.lower, self.upper)
        # Never permit a larger jump because the control loop ran late.
        maximum_delta = self.rate*min(DT, elapsed)
        target = np.clip(desired, self.previous_target-maximum_delta, self.previous_target+maximum_delta)
        return self._command(target, snapshot, now)

    def policy_command(self, action, snapshot, now):
        action = vector(action, N, 'policy action').clip(-1., 1.)
        desired = np.clip(self.nominal+self.scale*action, self.policy_lower, self.policy_upper)
        return self.target_command(desired, snapshot, now)

    def commit_sent(self, command, sent_time):
        """Call ONLY after publish succeeds. This is local publication, not motor ACK."""
        if (not np.isfinite(sent_time) or sent_time < command.observation_received
                or (self.last_sent_time is not None and sent_time < self.last_sent_time)):
            raise SafetyFault('Invalid/nonmonotonic publication timestamp')
        if sent_time > command.deadline:
            raise SafetyFault('Publish completed after command deadline')
        self.previous_target = command.model_q.copy()
        self.previous_action = command.effective_action.copy()
        self.last_sent_time = sent_time


def prepare_target(config, elapsed):
    previous = vector(config['prepare']['start_model_q'], N, 'start pose')
    for waypoint in config['prepare']['waypoints']:
        duration = waypoint['duration_s']
        target = vector(waypoint['model_q'], N, 'waypoint')
        if elapsed < duration:
            u = max(0., elapsed/duration)
            return previous+(target-previous)*(3*u*u-2*u*u*u), False
        elapsed -= duration
        previous = target
    return previous.copy(), True
