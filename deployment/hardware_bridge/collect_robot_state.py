"""Read-only ROS LowState probe; Python 3.10 + the robot's ROS packages only.

No motor-command publisher, Booster RPC client, model inference, or motor command is created.
ROS middleware may create its own discovery/parameter-event endpoints.
Callback timing is local receipt timing, not a sensor timestamp measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


def snapshot_message(message, received):
    issues = []

    def finite(value, label):
        try:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
            return value
        except (TypeError, ValueError, OverflowError):
            issues.append(label + ': missing/nonfinite number')
            return None

    def integer(value, label):
        number = finite(value, label)
        if number is not None and number.is_integer():
            return int(number)
        if number is not None:
            issues.append(label + ': expected integer')
        return None

    def motors(values, label):
        result = []
        for i, motor in enumerate(values):
            path = f'{label}[{i}]'
            row = {key: finite(getattr(motor, key, None), path+'.'+key)
                   for key in ('q', 'dq', 'ddq', 'tau_est')}
            row.update({key: integer(getattr(motor, key, None), path+'.'+key)
                        for key in ('mode', 'lost', 'temperature')})
            reserve = list(getattr(motor, 'reserve', ()))
            row['reserve'] = [integer(v, f'{path}.reserve[{j}]')
                              for j, v in enumerate(reserve)]
            if len(reserve) != 2:
                issues.append(f'{path}.reserve: expected 2 values, received {len(reserve)}')
            result.append(row)
        return result

    serial = motors(tuple(getattr(message, 'motor_state_serial', ())), 'serial')
    parallel = motors(tuple(getattr(message, 'motor_state_parallel', ())), 'parallel')
    imu = getattr(message, 'imu_state', None)
    vectors = {}
    for key in ('rpy', 'gyro', 'acc'):
        values = list(getattr(imu, key, ()))
        vectors[key] = [finite(v, f'imu.{key}[{i}]') for i, v in enumerate(values)]
        if len(values) != 3:
            issues.append(f'imu.{key}: expected 3 values, received {len(values)}')
    if len(serial) != 22:
        issues.append(f'serial count: expected 22 for this actor, received {len(serial)}')
    return {'received_monotonic_s': finite(received, 'receipt'),
            'serial_count': len(serial), 'parallel_count': len(parallel),
            'imu': vectors, 'motor_state_serial': serial, 'motor_state_parallel': parallel,
            'issues': issues}


def collect(seconds=5., initial_wait_s=10., topic='/low_state', clock=time.monotonic):
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError('seconds must be finite and in (0,60]')
    if not math.isfinite(initial_wait_s) or not 0 < initial_wait_s <= 10:
        raise ValueError('initial wait must be finite and in (0,10]')
    if not topic.startswith('/') or any(c.isspace() for c in topic):
        raise ValueError('topic must be an absolute ROS topic name')

    # Lazy imports: the module and its serialization tests need no ROS install.
    import rclpy
    from rclpy.context import Context
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from booster_interface.msg import LowState

    context = Context()
    node = None
    started = clock()
    first = last = last_recorded = None
    count = 0
    invalid_messages = 0
    gap_sum = gap_max = 0.
    serial_counts, parallel_counts = set(), set()
    samples, issues = [], set()
    completed = False
    stop_reason = None

    def callback(message):
        nonlocal first, last, last_recorded, count, invalid_messages, gap_sum, gap_max
        now = clock()
        sample = snapshot_message(message, now)
        if first is None:
            first = now
        if last is not None:
            gap = now-last['received_monotonic_s']
            gap_sum += gap
            gap_max = max(gap_max, gap)
        last = sample
        count += 1
        serial_counts.add(sample['serial_count']); parallel_counts.add(sample['parallel_count'])
        if sample['issues']:
            invalid_messages += 1
            issues.update(sample['issues'])
        # Keep JSON compact; all callbacks contribute to counts/cadence/faults.
        if last_recorded is None or now-last_recorded >= .05:
            samples.append(sample)
            last_recorded = now

    try:
        rclpy.init(args=[], context=context)
        node = rclpy.create_node('team15_read_only_state_probe', context=context,
                                 enable_rosout=False, start_parameter_services=False)
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(LowState, topic, callback, qos)
        while rclpy.ok(context=context):
            now = clock()
            if first is None and now-started >= initial_wait_s:
                stop_reason = 'no_state_before_initial_timeout'
                break
            if first is not None and now-first >= seconds:
                completed = True
                stop_reason = 'requested_interval_complete'
                break
            rclpy.spin_once(node, timeout_sec=.05)
        if stop_reason is None:
            stop_reason = 'ros_context_stopped'
    except KeyboardInterrupt:
        stop_reason = 'operator_interrupted'
    except Exception as exc:
        stop_reason = f'{type(exc).__name__}: {exc}'
    finally:
        if node is not None:
            node.destroy_node()
        context.try_shutdown()
    finished = clock()
    if last is not None and (not samples or samples[-1] is not last):
        samples.append(last)
    span = 0. if first is None or last is None else last['received_monotonic_s']-first
    last_age = None if last is None else finished-last['received_monotonic_s']
    # Recording a full wall-clock interval does not prove continuous reception.
    continuous = count >= 2 and last_age is not None and last_age <= .2 and gap_max <= .2
    summary = {
        'requested_seconds': seconds, 'completed_interval': completed, 'stop_reason': stop_reason,
        'callbacks_received': count, 'recorded_samples': len(samples),
        'serial_counts_seen': sorted(serial_counts), 'parallel_counts_seen': sorted(parallel_counts),
        'serial_count_matches_actor': serial_counts == {22},
        'invalid_messages': invalid_messages, 'issues': sorted(issues),
        'receipt_rate_hz': (count-1)/span if count > 1 and span > 0 else None,
        'receipt_gap_mean_s': gap_sum/(count-1) if count > 1 else None,
        'receipt_gap_max_s': gap_max if count > 1 else None,
        'last_sample_age_s': last_age,
        'receipt_continuity_under_0p2s': continuous,
        'last_rpy': None if last is None else last['imu']['rpy'],
        'last_serial_q': None if last is None else [m['q'] for m in last['motor_state_serial']],
        'passed_read_only_probe': bool(completed and continuous and count and not invalid_messages),
        'commands_sent': 0, 'mode_requests_sent': 0, 'hardware_control_validated': False,
    }
    return {'schema':'team15-read-only-robot-state-probe-v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'host':{'architecture':platform.machine(), 'python':platform.python_version(),
                    'ros_distro':os.environ.get('ROS_DISTRO'),
                    'ros_domain_id_environment':os.environ.get('ROS_DOMAIN_ID')},
            'topic':topic, 'qos':{'reliability':'best_effort','durability':'volatile','depth':1},
            'notes':['All cadence values are local callback receipt times, not sensor timestamps.',
                     'Raw motor fields are not calibrated or mapped to model joints by this probe.',
                     'JSON stores up to 20 Hz snapshots plus the final sample; all callbacks are counted.',
                     'The 0.2 s continuity flag is a diagnostic criterion, not a hardware watchdog setting.'],
            'summary':summary, 'samples':samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--initial-wait', type=float, default=10.)
    parser.add_argument('--topic', default='/low_state')
    parser.add_argument('--output', type=Path, default=Path('robot_state_probe.json'))
    args = parser.parse_args(argv)
    # Refuse overwriting before opening any ROS connection.
    with args.output.open('x', encoding='utf-8') as output:
        report = collect(args.seconds, args.initial_wait, args.topic)
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(report['summary'], indent=2, allow_nan=False))
    print(f'Saved: {args.output.resolve()}')
    return 0 if report['summary']['passed_read_only_probe'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as exc:
        print(f'Probe failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
