"""Interruptible, bounded real-control session; transport can be mocked.

Python timing/ROS publication is not hard real time or an actuator ACK.
The site must provide independent firmware deadman and physical stop/support.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np

from .core import DT, SafetyFault, prepare_target


class Session:
    def __init__(self, transport, core, actor, emit, clock=time.monotonic):
        self.transport, self.core, self.actor = transport, core, actor
        self.config, self.emit, self.clock = core.config, emit, clock
        self.stop_event = threading.Event()
        self.arm_event = threading.Event()
        self.run_event = threading.Event()
        self.stage = 'WAITING'
        self.reason = None
        self.exit_confirmed = None
        self.mode_thread = None
        self.mode_result = None
        self.mode_started = None
        self.stage_started = clock()
        self.last_command_at = None
        self.last_command = None
        self.motor_enabled = False
        self._stop_lock = threading.Lock()
        self._exit_lock = threading.Lock()
        self.exit_thread = None
        self.ready_since = None
        self.external_mode = None

    def request_stop(self, reason='operator stop'):
        with self._stop_lock:
            if self.stop_event.is_set():
                return
            self.reason = str(reason)
            self.stop_event.set()
            self.transport.inhibit_commands(self.reason)
        if self.motor_enabled:
            self._start_exit()

    def _start_exit(self):
        with self._exit_lock:
            if self.external_mode is not None:
                return
            if self.exit_thread is not None:
                return
            self.exit_thread = threading.Thread(target=self._exit_mode, daemon=True, name='team15-exit')
            self.exit_thread.start()

    def _exit_mode(self):
        # Independent from inference: a blocked actor cannot postpone this attempt.
        if self.mode_thread is not None:
            self.mode_thread.join(timeout=self.config['limits']['mode_timeout_s']+.2)
        try:
            if self.mode_thread is not None and self.mode_thread.is_alive():
                raise SafetyFault('Custom RPC did not terminate; use the physical stop')
            if self.external_mode is not None:
                return
            self.transport.change_mode(self.config['stop']['exit_mode'],
                                       timeout_s=self.config['limits']['mode_timeout_s'])
            self.exit_confirmed = True
        except Exception as exc:
            self.exit_confirmed = False
            self.emit({'event':'EXIT_UNCONFIRMED', 'error':str(exc),
                       'action':'Use the TA-reviewed physical stop/support procedure'})

    def request_arm(self, robot_serial):
        if self.stage != 'WAITING' or self.stop_event.is_set():
            raise SafetyFault('ARM accepted only once from WAITING')
        if robot_serial != self.config['review']['robot_serial']:
            raise SafetyFault('Robot serial does not match the reviewed site profile')
        self.arm_event.set()

    def request_run(self):
        if self.stage != 'READY' or self.stop_event.is_set():
            raise SafetyFault('RUN requires completed, held preparation pose')
        self.run_event.set()

    def _stage(self, value):
        self.stage = value
        self.stage_started = self.clock()
        self.emit({'event': 'stage', 'stage': value, 'time': self.stage_started})

    def _publish(self, command):
        if self.stop_event.is_set():
            raise SafetyFault('Session stopped before publish')
        now = self.clock()
        # Recheck the exact sample used by inference, not just the newest sample.
        if now > command.deadline or now-command.observation_received > self.config['limits']['state_timeout_s']:
            raise SafetyFault('State/command expired before publish')
        if self.transport.fault:
            raise SafetyFault(self.transport.fault)
        self.transport.publish(command.serial_q, command.kp, command.kd, deadline=command.deadline)
        sent = self.clock()
        self.core.commit_sent(command, sent)
        self.last_command_at = sent
        self.last_command = command
        self.emit({'event': 'command_published', 'time': sent, 'stage': self.stage,
                   'state_sequence': command.observation_sequence,
                   'model_q': command.model_q.tolist(), 'serial_q': command.serial_q.tolist(),
                   'effective_previous_action': command.effective_action.tolist()})

    def _begin_custom(self):
        if self.stop_event.is_set():
            return
        self.mode_started = self.clock()
        def change():
            try:
                self.transport.change_mode('custom', abort_event=self.stop_event,
                                           timeout_s=self.config['limits']['mode_timeout_s'])
                self.mode_result = True
            except Exception as exc:
                self.mode_result = str(exc)
                self.request_stop('Custom mode failed: '+str(exc))
        self.mode_thread = threading.Thread(target=change, daemon=True, name='team15-custom-rpc')
        self.mode_thread.start()

    def tick(self):
        if self.stop_event.is_set():
            return
        now = self.clock()
        snapshot = self.transport.latest_state()
        if self.transport.fault:
            raise SafetyFault(self.transport.fault)
        if self.stage == 'WAITING':
            if self.arm_event.is_set():
                command = self.core.prime(snapshot, now)
                self.transport.enable_motor_interface()
                self.transport.wait_for_command_subscriber(abort_event=self.stop_event,
                    timeout_s=self.config['limits']['subscriber_timeout_s'])
                # No commands/mode were issued during the bounded subscriber wait.
                command = self.core.prime(self.transport.latest_state(), self.clock())
                self.motor_enabled = True
                self._publish(command)  # Seed current pose before requesting Custom.
                self._stage('ARMING')
            return
        self.core.state(snapshot, now)
        self.emit({'event':'sensor_sample', 'time':now, 'sequence':snapshot.sequence,
                   'received_monotonic':snapshot.received_monotonic,
                   'q':list(snapshot.q), 'dq':list(snapshot.dq), 'tau':list(snapshot.tau),
                   'rpy':list(snapshot.rpy), 'gyro':list(snapshot.gyro)})
        if self.stage == 'ARMING':
            command = self.core.target_command(self.core.previous_target, snapshot, now)
            self._publish(command)
            if self.mode_thread is None and now-self.stage_started >= self.config['limits']['prime_settle_s']:
                self._begin_custom()
            if self.mode_result is True and not self.stop_event.is_set():
                self._stage('PREPARING')
            elif self.mode_started is not None and now-self.mode_started > self.config['limits']['mode_timeout_s']:
                raise SafetyFault('Custom mode confirmation timeout')
        elif self.stage == 'PREPARING':
            elapsed = now-self.stage_started
            if elapsed > self.config['limits']['prepare_timeout_s']:
                raise SafetyFault('Preparation timed out')
            desired, complete = prepare_target(self.config, elapsed)
            self._publish(self.core.target_command(desired, snapshot, now))
            if complete and np.max(np.abs(self.core.previous_target-self.core.nominal)) < 1e-6:
                q, dq, *_ = self.core.state(snapshot, self.clock())
                settled = (np.max(np.abs(q-self.core.nominal)) <= self.config['prepare']['ready_tolerance_rad']
                           and np.max(np.abs(dq)) <= self.config['prepare']['ready_speed_limit_rad_s'])
                if settled:
                    if self.ready_since is None:
                        self.ready_since = now
                    if now-self.ready_since >= self.config['prepare']['ready_dwell_s']:
                        self._stage('READY')
                else:
                    self.ready_since = None
        elif self.stage == 'READY':
            if now-self.stage_started > self.config['limits']['ready_timeout_s']:
                raise SafetyFault('Waiting-for-load timeout')
            self._publish(self.core.target_command(self.core.nominal, snapshot, now))
            if self.run_event.is_set():
                q, dq, *_ = self.core.state(snapshot, self.clock())
                if (np.max(np.abs(q-self.core.nominal)) > self.config['prepare']['ready_tolerance_rad']
                        or np.max(np.abs(dq)) > self.config['prepare']['ready_speed_limit_rad_s']):
                    raise SafetyFault('Robot moved away from the reviewed ready pose')
                self._stage('RUNNING')
        elif self.stage == 'RUNNING':
            if now-self.stage_started >= self.config['limits']['run_timeout_s']:
                self.request_stop('Requested run duration complete')
                return
            obs = self.core.observation(snapshot, now)
            started = self.clock()
            action = self.actor(obs)
            finished = self.clock()
            if finished-started > self.config['limits']['inference_timeout_s']:
                raise SafetyFault('Actor inference exceeded reviewed latency budget')
            if self.stop_event.is_set():
                return
            self._publish(self.core.policy_command(action, snapshot, finished))
            self.emit({'event':'actor_inference', 'time':finished, 'duration_s':finished-started,
                       'state_sequence':snapshot.sequence, 'obs':obs.tolist(),
                       'action':np.asarray(action).tolist()})

    def watchdog(self):
        """Independent from inference, but not a substitute for a firmware deadman."""
        while not self.stop_event.wait(.005):
            if not self.motor_enabled:
                continue
            now = self.clock()
            state = self.transport.latest_state()
            fault = self.transport.fault
            if fault:
                self.request_stop('Transport fault: '+str(fault))
            elif state is None or now-state.received_monotonic > self.config['limits']['state_timeout_s']:
                self.request_stop('State watchdog expired')
            elif self.last_command_at is not None and now-self.last_command_at > self.config['limits']['command_timeout_s']:
                self.request_stop('Command watchdog expired')

    def run(self):
        monitor = threading.Thread(target=self.watchdog, daemon=True, name='team15-watchdog')
        monitor.start()
        ownership = threading.Thread(target=self.monitor_mode, daemon=True, name='team15-mode-monitor')
        ownership.start()
        try:
            while not self.stop_event.is_set():
                start = self.clock()
                try:
                    self.tick()
                except Exception as exc:
                    self.request_stop(str(exc))
                self.stop_event.wait(max(0., DT-(self.clock()-start)))
        finally:
            self.finish()
            monitor.join(timeout=.1)
            ownership.join(timeout=self.config['limits']['status_rpc_timeout_s']+.2)

    def monitor_mode(self):
        while not self.stop_event.wait(self.config['limits']['mode_check_interval_s']):
            if self.stage not in ('PREPARING', 'READY', 'RUNNING'):
                continue
            try:
                mode = self.transport.get_mode(timeout_s=self.config['limits']['status_rpc_timeout_s'])
                if mode != 3:
                    # Respect an external controller/physical stop; never turn
                    # somebody else's Damping stop back into Walking.
                    self.external_mode = mode
                    self.request_stop(f'Robot left Custom mode (mode={mode}); external mode preserved')
                    return
            except Exception as exc:
                self.request_stop('Mode monitor: '+str(exc))

    def finish(self):
        self.request_stop(self.reason or 'session end')
        self.stage = 'STOPPED'
        if self.motor_enabled:
            self._start_exit()
            if self.exit_thread is not None:
                self.exit_thread.join(timeout=2*self.config['limits']['mode_timeout_s']+.5)
                if self.exit_thread.is_alive():
                    self.exit_confirmed = False
                    self.emit({'event':'EXIT_UNCONFIRMED', 'error':'Exit worker exceeded bounded wait'})
        self.emit({'event':'session_end', 'reason':self.reason,
                   'commands_published':self.transport.commands_sent,
                   'exit_mode_confirmed':self.exit_confirmed,
                   'external_mode_preserved':self.external_mode,
                   'hardware_balance_validated':False})
