"""Run preflight, read-only shadow, or an explicitly armed site-configured session."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time

from team15_hardware.core import Actor, ControlCore, SafetyFault
from team15_hardware.config import load_config

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / 'team15_proprio_package/tasks/team15_proprio/models/proprio_hold.onnx'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight', action='store_true', help='Validate config/model only; no ROS connection (default)')
    modes.add_argument('--shadow', action='store_true', help='ROS read + inference only; no publisher or mode RPC')
    modes.add_argument('--live', action='store_true', help='Enable operator-armed real control after complete site checks')
    parser.add_argument('--seconds', type=float, default=20., help='Shadow duration or upper bound on active policy duration')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    contract = json.loads(args.model.with_name('proprio_hold.contract.json').read_text(encoding='utf-8'))
    config = load_config(args.config, contract)  # Must fail before ROS/network setup.
    if args.live and config['deployment_target'] != 'real_robot':
        raise SafetyFault('Simulation fixture cannot be used with --live')
    actor = Actor(args.model)
    vectors = actor.verify_vectors(args.model.with_name('proprio_hold.test_vectors.npz'))
    print(json.dumps({'preflight':'passed', **vectors,
                      'hardware_validated':False, 'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest()}))
    if not (args.live or args.shadow):
        return 0
    if not 0 < args.seconds <= 600:
        raise SafetyFault('--seconds must be in (0,600]')
    if args.output is None:
        raise SafetyFault('--output NEW_DIRECTORY is required for ROS sessions')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'site_profile.json').write_bytes(args.config.read_bytes())
    config = deepcopy(config)
    if args.live:
        config['limits']['run_timeout_s'] = min(args.seconds, config['limits']['run_timeout_s'])
    (args.output/'effective_profile.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log = (args.output/'events.jsonl').open('x', encoding='utf-8', buffering=1)
    log_lock = threading.Lock()
    def emit(row):
        with log_lock:
            log.write(json.dumps(row, allow_nan=False)+'\n')
        if row.get('event') in ('stage', 'session_end', 'EXIT_UNCONFIRMED', 'shadow_complete'):
            print(json.dumps(row), flush=True)
    from team15_hardware.ros_transport import RosTransport
    from team15_hardware.runtime import Session
    transport = RosTransport(config['transport'])
    session = None
    try:
        transport.start()
        core = ControlCore(config, actor.contract)
        if args.shadow:
            deadline = time.monotonic()+args.seconds
            count = 0
            while time.monotonic() < deadline:
                if transport.fault:
                    raise SafetyFault(transport.fault)
                state = transport.latest_state()
                if state is not None:
                    obs = core.observation(state, time.monotonic())
                    action = actor(obs)
                    emit({'event':'shadow', 'sequence':state.sequence, 'obs':obs.tolist(),
                          'candidate_action':action.tolist(), 'previous_action':'zero cold start', 'commands_sent':0})
                    count += 1
                time.sleep(.02)
            if count == 0:
                raise SafetyFault('No robot samples received during shadow interval')
            emit({'event':'shadow_complete', 'samples':count, 'commands_sent':transport.commands_sent,
                  'closed_loop_validated':False})
        else:
            session = Session(transport, core, actor, emit)
            def stop_signal(signum, frame):
                session.request_stop(f'operator signal {signum}')
            signal.signal(signal.SIGINT, stop_signal)
            signal.signal(signal.SIGTERM, stop_signal)
            def operator():
                print('Commands: arm ROBOT_SERIAL ; run (only after READY + TA-approved load/support); stop', flush=True)
                while not session.stop_event.is_set():
                    try:
                        command = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        session.request_stop('Operator console closed')
                        return
                    try:
                        if command.startswith('arm '): session.request_arm(command[4:].strip())
                        elif command == 'run': session.request_run()
                        elif command == 'stop': session.request_stop()
                        else: print('Expected arm ROBOT_SERIAL, run, or stop', flush=True)
                    except SafetyFault as exc:
                        print(str(exc), flush=True)
            threading.Thread(target=operator, daemon=True, name='team15-operator').start()
            session.run()
            if session.exit_confirmed is False:
                return 2
            if session.reason not in ('Requested run duration complete', 'operator stop'):
                return 1
    finally:
        transport.close()
        log.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(1)
