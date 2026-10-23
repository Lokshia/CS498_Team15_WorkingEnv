"""Viewer mass transactions without a browser, renderer, or listening server.

Set LIFTING_TEST_ENV=1 for three fresh CPU worlds. No test starts a viewer or
allocates CUDA tensors; lifecycle tests use plain stand-ins for GUI handles.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifting_lab import cli, mdp
from lifting_lab.model import PayloadCfg
from lifting_lab.viewer import (
    MassAdjustableViewer, RecordingViserPlayViewer, prepare_world,
    validate_viewer_mass, world_snapshot,
)


class ViewerMassValidationTests(unittest.TestCase):
    def test_zero_and_upper_boundary_are_accepted(self):
        self.assertEqual(validate_viewer_mass(0), 0.0)
        self.assertEqual(validate_viewer_mass(10), 10.0)
        self.assertEqual(validate_viewer_mass("0.5"), 0.5)

    def test_negative_nonfinite_and_above_limit_are_rejected(self):
        for value in (-1, -.000001, np.nan, np.inf, -np.inf, 10.000001, 100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_viewer_mass(value)


class MassRequestTests(unittest.TestCase):
    def setUp(self):
        # Calling the method on this minimal stand-in cannot start Viser or open
        # a socket. _is_recording is normally the recorder's read-only property.
        self.viewer = SimpleNamespace(
            _lifecycle_lock=RLock(), _closing=False, _pending_mass=None,
            _is_recording=False, _record_operations=0,
            _mass_notice=SimpleNamespace(content="initial"),
            _weight_input=SimpleNamespace(disabled=False),
            _apply_mass_button=SimpleNamespace(disabled=False),
            _record_button=SimpleNamespace(disabled=False),
        )

    def request(self, value):
        return MassAdjustableViewer.request_mass(self.viewer, value)

    def test_valid_zero_request_is_queued_and_disables_conflicting_controls(self):
        self.assertTrue(self.request(0.0))
        self.assertEqual(self.viewer._pending_mass, 0.0)
        self.assertTrue(self.viewer._weight_input.disabled)
        self.assertTrue(self.viewer._apply_mass_button.disabled)
        self.assertTrue(self.viewer._record_button.disabled)
        self.assertIn("重建", self.viewer._mass_notice.content)

    def test_recording_rejects_mass_changes(self):
        self.viewer._is_recording = True
        self.assertFalse(self.request(2.0))
        self.assertIsNone(self.viewer._pending_mass)
        self.assertFalse(self.viewer._weight_input.disabled)
        self.assertIn("结束录像", self.viewer._mass_notice.content)

    def test_finishing_record_operation_rejects_mass_changes(self):
        self.viewer._record_operations = 1
        self.assertFalse(self.request(2.0))
        self.assertIsNone(self.viewer._pending_mass)

    def test_duplicate_request_does_not_replace_zero_mass_pending_request(self):
        self.assertTrue(self.request(0.0))
        self.assertFalse(self.request(2.0))
        self.assertEqual(self.viewer._pending_mass, 0.0)

    def test_closing_rejects_requests(self):
        self.viewer._closing = True
        self.assertFalse(self.request(.5))
        self.assertIsNone(self.viewer._pending_mass)

    def test_invalid_request_never_queues_or_disables_controls(self):
        for value in (-1, 10.1, np.nan, np.inf, None, "invalid"):
            with self.subTest(value=value):
                self.assertFalse(self.request(value))
                self.assertIsNone(self.viewer._pending_mass)
                self.assertFalse(self.viewer._weight_input.disabled)
                self.assertFalse(self.viewer._apply_mass_button.disabled)
                self.assertFalse(self.viewer._record_button.disabled)

    def test_pending_tick_pauses_and_discards_retiring_world_actions(self):
        self.assertTrue(self.request(2.0))
        self.viewer.pause = Mock()
        self.viewer.requested_mass_kg = None
        self.viewer._actions = ["play", "step"]
        self.viewer._interrupted = False
        self.assertFalse(MassAdjustableViewer.tick(self.viewer))
        self.viewer.pause.assert_called_once_with()
        self.assertEqual(self.viewer.requested_mass_kg, 2.0)
        self.assertEqual(self.viewer._actions, [])
        self.assertTrue(self.viewer._closing)
        self.assertTrue(self.viewer._interrupted)


class WorldPreparationCleanupTests(unittest.TestCase):
    def test_policy_failure_closes_partial_world(self):
        env = Mock()
        args = SimpleNamespace(seconds=8.0, seed=123)
        factory = Mock(return_value=(env, {"actual_payload_mass_kg": .5}))
        policy = Mock(side_effect=ValueError("fixture policy failure"))
        with self.assertRaisesRegex(ValueError, "fixture policy failure"):
            prepare_world(args, factory, policy)
        factory.assert_called_once_with(args, seconds=9.0, auto_reset=True, num_envs=1)
        env.close.assert_called_once_with()

    def test_wrapper_failure_closes_partial_world(self):
        env = Mock()
        args = SimpleNamespace(seconds=8.0, seed=123)
        with patch("lifting_lab.viewer.RslRlVecEnvWrapper",
                   side_effect=RuntimeError("fixture wrapper failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture wrapper failure"):
                prepare_world(args, Mock(return_value=(env, {})), Mock())
        env.close.assert_called_once_with()


class ViewerInitialSceneTests(unittest.TestCase):
    def test_paused_setup_publishes_initial_scene_once_without_stepping_physics(self):
        viewer = MassAdjustableViewer.__new__(MassAdjustableViewer)
        viewer.mass = {"requested_payload_mass_kg": .5, "actual_payload_mass_kg": .5}
        viewer.notice = ""
        viewer._is_paused = True
        data = object()
        raw_env = SimpleNamespace(sim=SimpleNamespace(data=data), step=Mock())
        viewer.env = SimpleNamespace(unwrapped=raw_env, step=Mock())
        server = MagicMock()
        server._client_lock = RLock()
        foreign_callback, owned_callback = object(), object()
        server._client_connect_cb = [foreign_callback]
        server._client_disconnect_cb = []
        viewer._server = server
        active = {"simulation": False, "atomic": False}
        events = []

        @contextmanager
        def guard(name):
            active[name] = True
            try:
                yield
            finally:
                active[name] = False

        viewer._sim_lock = guard("simulation")
        server.atomic.side_effect = lambda: guard("atomic")
        scene = Mock()

        def base_setup(instance):
            events.append("base_setup")
            instance._scene = scene
            server._client_connect_cb.append(owned_callback)

        def update(actual_data):
            self.assertIs(actual_data, data)
            self.assertTrue(active["simulation"])
            self.assertTrue(active["atomic"])
            self.assertIs(scene.paused, True)
            events.append("update")

        def flush():
            self.assertTrue(active["simulation"])
            self.assertTrue(active["atomic"])
            events.append("flush")

        scene.update.side_effect = update
        server.flush.side_effect = flush
        with patch.object(RecordingViserPlayViewer, "setup", autospec=True,
                          side_effect=base_setup) as setup:
            viewer.setup()
            setup.assert_called_once_with(viewer)
        scene.update.assert_called_once_with(data)
        server.flush.assert_called_once_with()
        server.atomic.assert_called_once_with()
        self.assertEqual(events, ["base_setup", "update", "flush"])
        self.assertEqual(active, {"simulation": False, "atomic": False})
        self.assertEqual(viewer._owned_connect_callbacks, [owned_callback])
        viewer.env.step.assert_not_called()
        raw_env.step.assert_not_called()


class ViewerLifecycleTests(unittest.TestCase):
    def setUp(self):
        # A real subclass instance is necessary for super(), but its constructor
        # must not run: it could otherwise allocate GUI/server resources.
        self.viewer = MassAdjustableViewer.__new__(MassAdjustableViewer)
        self.viewer._lifecycle_lock = RLock()
        self.viewer._closing = False
        self.viewer._closed = False
        self.viewer._pending_mass = None
        self.viewer._record_operations = 0
        self.viewer._lockstep = None
        self.viewer._record_thread = None
        self.viewer._threadpool = Mock()  # Resource exists, but no worker is started.
        self.foreign_connect, self.owned_connect = object(), object()
        self.foreign_disconnect, self.owned_disconnect = object(), object()
        self.viewer._owned_connect_callbacks = [self.owned_connect]
        self.viewer._owned_disconnect_callbacks = [self.owned_disconnect]
        self.viewer._server = SimpleNamespace(
            _client_lock=RLock(),
            _client_connect_cb=[self.foreign_connect, self.owned_connect],
            _client_disconnect_cb=[self.foreign_disconnect, self.owned_disconnect],
        )

    def test_recording_start_is_blocked_during_all_conflicting_states(self):
        states = (("_record_operations", 1), ("_lockstep", object()),
                  ("_closing", True), ("_pending_mass", 0.0))
        for name, value in states:
            with self.subTest(state=name):
                original = getattr(self.viewer, name)
                setattr(self.viewer, name, value)
                try:
                    with patch.object(RecordingViserPlayViewer, "_start_recording",
                                      autospec=True) as base_start:
                        self.viewer._start_recording(object())
                        base_start.assert_not_called()
                finally:
                    setattr(self.viewer, name, original)

    def test_valid_recording_start_calls_base_exactly_once(self):
        client, result = object(), object()
        with patch.object(RecordingViserPlayViewer, "_start_recording",
                          autospec=True, return_value=result) as base_start:
            self.assertIs(self.viewer._start_recording(client), result)
            base_start.assert_called_once_with(self.viewer, client)

    def test_close_is_idempotent_and_preserves_foreign_callbacks(self):
        with patch.object(RecordingViserPlayViewer, "close", autospec=True) as base_close:
            self.viewer.close()
            self.viewer.close()
            base_close.assert_called_once_with(self.viewer)
        self.assertTrue(self.viewer._closing)
        self.assertTrue(self.viewer._closed)
        self.assertEqual(self.viewer._server._client_connect_cb, [self.foreign_connect])
        self.assertEqual(self.viewer._server._client_disconnect_cb, [self.foreign_disconnect])

    def test_setup_exception_still_closes_once_without_running_a_tick(self):
        self.viewer.setup = Mock(side_effect=RuntimeError("fixture setup failure"))
        self.viewer.tick = Mock()
        with patch.object(RecordingViserPlayViewer, "close", autospec=True) as base_close:
            with patch.object(self.viewer, "close", wraps=self.viewer.close) as close:
                with self.assertRaisesRegex(RuntimeError, "fixture setup failure"):
                    self.viewer.run(num_steps=1)
                close.assert_called_once_with()
            base_close.assert_called_once_with(self.viewer)
        self.viewer.setup.assert_called_once_with()
        self.viewer.tick.assert_not_called()
        self.assertTrue(self.viewer._closed)

    def test_early_setup_failure_without_threadpool_still_removes_owned_callbacks(self):
        del self.viewer._threadpool
        self.viewer.setup = Mock(side_effect=RuntimeError("fixture early setup failure"))
        with patch.object(RecordingViserPlayViewer, "close", autospec=True) as base_close:
            with self.assertRaisesRegex(RuntimeError, "fixture early setup failure"):
                self.viewer.run(num_steps=1)
            self.viewer.close()
            base_close.assert_not_called()
        self.assertTrue(self.viewer._closed)
        self.assertEqual(self.viewer._server._client_connect_cb, [self.foreign_connect])
        self.assertEqual(self.viewer._server._client_disconnect_cb, [self.foreign_disconnect])


@unittest.skipUnless(os.environ.get("LIFTING_TEST_ENV") == "1",
                     "Set LIFTING_TEST_ENV=1 for fresh CPU world checks")
class FreshCpuWorldTests(unittest.TestCase):
    def test_rebuilds_mass_and_inertia_with_zero_time_aligned_grips_and_97_observations(self):
        snapshots = []
        # A regression that accidentally starts the server must fail, not bind a
        # port; prepare_world itself should only build simulation and wrappers.
        with patch("lifting_lab.viewer.viser.ViserServer",
                   side_effect=AssertionError("Tests must not start a viewer server")):
            for revision, requested in enumerate((.5, 2.0, 0.0)):
                with self.subTest(requested_mass_kg=requested):
                    args = cli.build_parser().parse_args([
                        "demo", "--device", "cpu", "--mass", str(requested),
                        "--seconds", "2", "--seed", "123", "--threads", "2",
                        "--support-mode", "pregrasp",
                    ])
                    env, mass, policy, wrapper = prepare_world(args, cli._env, cli._policy)
                    try:
                        snapshot = world_snapshot(env, mass, revision=revision, paused=True)
                        snapshots.append(snapshot)
                        actual = max(requested, .001)
                        self.assertEqual(env.device, "cpu")
                        self.assertEqual(env.num_envs, 1)
                        self.assertEqual(snapshot["revision"], revision)
                        self.assertTrue(snapshot["paused"])
                        self.assertEqual(snapshot["mass"]["requested_payload_mass_kg"], requested)
                        self.assertEqual(snapshot["mass"]["actual_payload_mass_kg"], actual)
                        self.assertAlmostEqual(snapshot["compiled_payload_mass_kg"], actual)
                        self.assertAlmostEqual(snapshot["warp_payload_mass_kg"], actual, places=7)
                        np.testing.assert_allclose(snapshot["compiled_payload_inertia_kg_m2"],
                                                   PayloadCfg(mass_kg=requested).diagonal_inertia,
                                                   rtol=1e-7, atol=1e-10)
                        self.assertEqual(snapshot["simulation_time_s"], 0.0)
                        self.assertEqual(snapshot["control_steps"], 0)
                        self.assertTrue(torch.all(env.sim.data.time == 0).item())
                        grip_error = torch.linalg.vector_norm(mdp.grip_displacement(env), dim=-1)
                        self.assertLess(grip_error.max().item(), 2e-6)
                        obs = wrapper.get_observations()
                        for name in ("actor", "critic"):
                            self.assertEqual(tuple(obs[name].shape), (1, 97))
                            self.assertTrue(torch.isfinite(obs[name]).all().item())
                        self.assertEqual(tuple(policy(obs).shape), (1, 22))
                    finally:
                        env.close()
        inertias = [np.asarray(row["compiled_payload_inertia_kg_m2"]) for row in snapshots]
        np.testing.assert_allclose(inertias[1], 4 * inertias[0], rtol=1e-10)
        np.testing.assert_allclose(inertias[2], .002 * inertias[0], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
