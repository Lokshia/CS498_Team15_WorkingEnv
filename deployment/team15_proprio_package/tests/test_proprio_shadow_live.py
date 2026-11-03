"""No ROS installation or robot required: fake-clock live/error-path tests.

This file also works from a distributed package's tests/ directory. Only numpy
and the Python standard library are needed; inference is stubbed for CLI tests.
"""
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deployment/booster_proprio"
if not (PACKAGE / "scripts/shadow_check.py").is_file():
    PACKAGE = ROOT


def load_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, PACKAGE / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow = load_script("shadow_check.py", "proprio_shadow_live_tests")
installer = load_script("install_task.py", "proprio_installer_live_tests")


def message(count=22):
    return SimpleNamespace(
        imu_state=SimpleNamespace(rpy=[0., 0., 0.], gyro=[0., 0., 0.]),
        motor_state_serial=[SimpleNamespace(q=0., dq=0., tau_est=1.) for _ in range(count)],
    )


def args(**overrides):
    result = dict(seconds=.12, startup_timeout=.04, max_age=.025, topic="/low_state")
    result.update(overrides)
    return SimpleNamespace(**result)


class FakeROS:
    """spin_once advances a fake clock; it never sleeps or touches a network."""
    def __init__(self, events=None, always_fresh=False):
        self.now = 0.
        self.running = True
        self.events = list(events or [])
        self.always_fresh = always_fresh
        self.calls = []
        self.callback = None
        self.destroyed = False
        self.shutdown_count = 0
        self.spin_count = 0
        self.node = SimpleNamespace(
            create_subscription=self.subscribe,
            create_publisher=self.forbidden,
            create_client=self.forbidden,
            create_service=self.forbidden,
            destroy_node=self.destroy,
        )
        ros = ModuleType("rclpy")
        ros.init = lambda **kw: self.calls.append(("init", kw))
        ros.create_node = lambda name: self.node
        ros.spin_once = self.spin
        ros.ok = lambda: self.running
        ros.shutdown = self.shutdown
        qos = ModuleType("rclpy.qos")
        qos.QoSProfile = lambda **kw: SimpleNamespace(**kw)
        qos.ReliabilityPolicy = SimpleNamespace(BEST_EFFORT="best_effort")
        qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")
        msgs = ModuleType("booster_interface.msg")
        msgs.LowState = type("LowState", (), {})
        self.modules = {"rclpy": ros, "rclpy.qos": qos,
                        "booster_interface": ModuleType("booster_interface"),
                        "booster_interface.msg": msgs}

    def forbidden(self, *unused, **kwargs):
        raise AssertionError("A command/service communication API was called")

    def subscribe(self, kind, topic, callback, qos):
        self.callback = callback
        self.calls.append(("subscribe", kind, topic, qos))

    def spin(self, node, timeout_sec):
        self.spin_count += 1
        if self.spin_count > 1000:
            raise AssertionError("Live diagnostic did not obey its deadline")
        self.now += timeout_sec
        if self.events:
            event = self.events.pop(0)
            if event == "stop":
                self.running = False
            elif event is not None:
                self.callback(event)
        elif self.always_fresh:
            self.callback(message())

    def destroy(self):
        self.destroyed = True

    def shutdown(self):
        self.shutdown_count += 1

    @contextmanager
    def active(self):
        fake_time = SimpleNamespace(monotonic=lambda: self.now, perf_counter=lambda: self.now)
        with patch.dict(sys.modules, self.modules), patch.object(shadow, "time", fake_time):
            yield self


class FakeInference:
    contract = {"onnx_sha256": "synthetic-test-only",
                "joint_mapping": {"booster_deploy_joint_names": [f"joint{i}" for i in range(22)]}}

    def infer(self, record):
        obs = shadow.observation(record, np.zeros(22, np.float32))
        return dict(observation=obs.tolist(), proposed_action=[0.] * 22,
                    proposed_unprojected_target_rad=[0.] * 22,
                    inference_ms=.1, commands_sent=0,
                    previous_action_mode="zero_each_frame_cold_start_only")


class LiveTests(unittest.TestCase):
    def run_live(self, ros, options=None):
        emitted, stats = [], {"received_messages": 0, "completed_requested_duration": False}
        with ros.active():
            shadow.run_live(options or args(), lambda record, **kw: emitted.append((record, kw)), stats)
        return emitted, stats

    def assert_cleanup(self, ros):
        self.assertTrue(ros.destroyed)
        self.assertEqual(ros.shutdown_count, 1)

    def test_live_fresh_stream_completes_without_command_apis(self):
        ros = FakeROS(always_fresh=True)
        emitted, stats = self.run_live(ros)
        self.assertGreater(len(emitted), 2)
        self.assertTrue(stats["completed_requested_duration"])
        self.assertGreaterEqual(stats["actual_elapsed_s"], .12)
        self.assertLessEqual(stats["actual_elapsed_s"], .14)
        sequences = [extra["sequence"] for _, extra in emitted]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertTrue(all(extra["receive_age_s"] == 0 for _, extra in emitted))
        subscription = next(c for c in ros.calls if c[0] == "subscribe")
        self.assertEqual(subscription[2], "/low_state")
        self.assertEqual(subscription[3].depth, 1)
        self.assertEqual(subscription[3].reliability, "best_effort")
        self.assertEqual(subscription[3].history, "keep_last")
        self.assert_cleanup(ros)

    def test_startup_timeout_cleans_up(self):
        ros = FakeROS()
        with self.assertRaisesRegex(TimeoutError, "No LowState arrived"):
            self.run_live(ros)
        self.assertLess(ros.now, .07)
        self.assert_cleanup(ros)

    def test_stalled_stream_does_not_reuse_last_sample(self):
        ros = FakeROS(events=[message()])
        emitted, stats = [], {"received_messages": 0}
        with ros.active(), self.assertRaisesRegex(TimeoutError, "reception stalled"):
            shadow.run_live(args(), lambda record, **kw: emitted.append(record), stats)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(stats["received_messages"], 1)
        self.assert_cleanup(ros)

    def test_bad_joint_count_is_terminal(self):
        ros = FakeROS(events=[message(21)], always_fresh=True)
        with self.assertRaisesRegex(ValueError, "Invalid LowState.*22 serial joints"):
            self.run_live(ros)
        self.assertEqual(ros.spin_count, 1)
        self.assert_cleanup(ros)

    def test_nonfinite_imu_after_good_sample_is_terminal(self):
        bad = message()
        bad.imu_state.gyro[0] = float("nan")
        ros = FakeROS(events=[message(), bad], always_fresh=True)
        with self.assertRaisesRegex(ValueError, "Invalid LowState.*gyro.*finite"):
            self.run_live(ros)
        self.assertEqual(ros.spin_count, 2)
        self.assert_cleanup(ros)

    def test_early_ros_shutdown_is_not_a_completed_capture(self):
        ros = FakeROS(events=[message(), "stop"])
        with self.assertRaisesRegex(RuntimeError, "ROS stopped before.*duration"):
            self.run_live(ros)
        self.assertLess(ros.now, .12)
        self.assert_cleanup(ros)

    def test_subscription_setup_failure_cleans_up(self):
        ros = FakeROS()
        ros.node.create_subscription = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad topic"))
        with self.assertRaisesRegex(RuntimeError, "bad topic"):
            self.run_live(ros)
        self.assert_cleanup(ros)

    def test_slow_inference_crossing_deadline_is_failed_not_complete(self):
        ros = FakeROS(events=[message()])

        class SlowInference(FakeInference):
            def infer(self, record):
                ros.now += 1.
                result = super().infer(record)
                result["inference_ms"] = 1000.
                return result

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            argv = ["shadow_check.py", "--output", str(output), "--seconds", ".12", "--max-age", ".025"]
            with ros.active(), patch.object(sys, "argv", argv), \
                    patch.object(shadow, "ShadowInference", SlowInference), redirect_stdout(io.StringIO()):
                result = shadow.main()
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertFalse(report["passed"])
            self.assertFalse(report["completed_requested_duration"])
            self.assertGreater(report["actual_elapsed_s"], report["requested_duration_s"])
            self.assertEqual(report["samples"], 1)
            self.assertIn("Inference/recording stalled", report["error"])
            self.assertEqual(report["commands_sent"], 0)
            self.assert_cleanup(ros)

    def test_main_records_early_shutdown_as_failed_with_prefix_samples(self):
        ros = FakeROS(events=[message(), "stop"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            argv = ["shadow_check.py", "--output", str(output), "--seconds", ".12"]
            with ros.active(), patch.object(sys, "argv", argv), \
                    patch.object(shadow, "ShadowInference", FakeInference), redirect_stdout(io.StringIO()):
                result = shadow.main()
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertFalse(report["passed"])
            self.assertFalse(report["completed_requested_duration"])
            self.assertEqual(report["requested_duration_s"], .12)
            self.assertEqual(report["samples"], 1)
            self.assertEqual(report["source"], "live_ros2")
            self.assertEqual(report["commands_sent"], 0)
            self.assertFalse(report["hardware_control_tested"])
            self.assertIn("ROS stopped", report["error"])


class ReplayTests(unittest.TestCase):
    def replay(self, lines):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text("\n".join(lines), encoding="utf-8")
            output = root / "replay"
            argv = ["shadow_check.py", "--output", str(output), "--replay-jsonl", str(source)]
            with patch.object(sys, "argv", argv), patch.object(shadow, "ShadowInference", FakeInference), \
                    redirect_stdout(io.StringIO()):
                result = shadow.main()
            report = json.loads((output / "summary.json").read_text())
            samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
            return result, report, samples

    def test_empty_replay_fails(self):
        result, report, samples = self.replay([])
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        self.assertIn("No diagnostic samples", report["error"])
        self.assertEqual(samples, [])

    def test_malformed_json_after_valid_sample_fails_without_losing_evidence(self):
        valid = json.dumps(shadow.low_state_record(message()))
        result, report, samples = self.replay([valid, "{not-json"])
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        self.assertIn("JSONDecodeError", report["error"])
        self.assertEqual(report["samples"], 1)
        self.assertEqual(len(samples), 1)

    def test_bad_vector_replay_fails(self):
        record = shadow.low_state_record(message())
        record["q"] = [0.] * 21
        result, report, samples = self.replay([json.dumps(record)])
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        self.assertIn("22 finite", report["error"])
        self.assertEqual(samples, [])

    def test_replay_success_is_explicitly_not_hardware_or_duration_test(self):
        record = shadow.low_state_record(message())
        result, report, samples = self.replay([json.dumps({"state": record})] * 3)
        self.assertEqual(result, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["source"], "replay")
        self.assertIsNone(report["requested_duration_s"])
        self.assertIsNone(report["completed_requested_duration"])
        self.assertFalse(report["hardware_control_tested"])
        for sample in samples:
            self.assertEqual(sample["commands_sent"], 0)
            self.assertEqual(sample["observation"][50:], [0.] * 22)


class InstallerTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package, repo = root / "package", root / "repo"
            task = package / "tasks/team15_proprio"
            task.mkdir(parents=True)
            (task / "__init__.py").write_text("# isolated task\n")
            (task / "simulate.py").write_text("# simulation only\n")
            manifest = {}
            for f in task.iterdir():
                manifest[f.relative_to(package).as_posix()] = dict(
                    sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)
            (package / "SHA256SUMS.json").write_text(json.dumps(manifest))
            for relative in ["scripts/deploy.py", "booster_deploy/controllers/mujoco_controller.py", "tasks/__init__.py"]:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# official fixture\n")

            def git_output(cmd, **kwargs):
                if cmd[-1] == "--show-toplevel":
                    return str(repo) + "\n"
                if cmd[-1] == "HEAD":
                    return installer.DEPLOY_COMMIT + "\n"
                if "diff" in cmd:
                    return ""
                raise AssertionError(cmd)

            with patch.object(installer, "PACKAGE", package), \
                    patch.object(installer.subprocess, "check_output", side_effect=git_output):
                yield package, repo

    def test_nested_package_directory_is_rejected(self):
        with self.fixture() as (_, repo):
            with self.assertRaisesRegex(ValueError, "checkout top-level"):
                installer.install(repo / "booster_deploy")
            self.assertFalse((repo / "booster_deploy/tasks").exists())

    def test_install_is_idempotent_and_does_not_enable_hardware(self):
        with self.fixture() as (_, repo):
            first, second = installer.install(repo), installer.install(repo)
            self.assertEqual(first, second)
            self.assertEqual(first["task_files"], 2)
            self.assertFalse(first["hardware_control_enabled"])
            self.assertTrue((repo / "tasks/team15_proprio/simulate.py").is_file())

    def test_different_existing_task_file_is_not_overwritten(self):
        with self.fixture() as (_, repo):
            existing = repo / "tasks/team15_proprio/simulate.py"
            existing.parent.mkdir(parents=True)
            existing.write_text("# TA local change\n")
            with self.assertRaises(FileExistsError):
                installer.install(repo)
            self.assertEqual(existing.read_text(), "# TA local change\n")
            self.assertFalse((existing.parent / "__init__.py").exists())

    def test_changed_package_rejected_before_copy(self):
        with self.fixture() as (package, repo):
            (package / "tasks/team15_proprio/simulate.py").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Changed package file"):
                installer.install(repo)
            self.assertFalse((repo / "tasks/team15_proprio").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
