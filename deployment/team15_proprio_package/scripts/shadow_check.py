"""Read-only K1 sensor/inference diagnostics. Never publishes motor commands.

Each observation uses ZERO previous action: a cold-start diagnostic, not a
closed-loop policy rollout. No BoosterRobotPortal, mode RPC or command topic
is imported/created. ROS imports occur only in live mode; JSONL replay works
without ROS and does not establish hardware compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGE / "tasks/team15_proprio/models"
EXPECTED_CHECKPOINT = "b0722baf7d79d513d54cb8afec74860ac4aed5296bbd8049d73ec9e4afcc1382"
EXPECTED_CONTRACT = "c22b6e5138c5ce085afebab39304cf4649e02d1bd39b78bd15c0f9a50107e2d7"


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return result


def low_state_record(message):
    motors = message.motor_state_serial
    if len(motors) != 22:
        raise ValueError(f"Expected K1 22 serial joints; received {len(motors)}")
    return {
        "rpy": vector(message.imu_state.rpy, 3, "imu_state.rpy").tolist(),
        "gyro": vector(message.imu_state.gyro, 3, "imu_state.gyro").tolist(),
        "q": vector([m.q for m in motors], 22, "motor q").tolist(),
        "dq": vector([m.dq for m in motors], 22, "motor dq").tolist(),
        "tau_est": vector([m.tau_est for m in motors], 22, "motor tau_est").tolist(),
    }


def observation(record, default):
    roll, pitch, _ = vector(record["rpy"], 3, "rpy (radians)")
    gyro = vector(record["gyro"], 3, "body gyro (rad/s)")
    q, dq = vector(record["q"], 22, "q"), vector(record["dq"], 22, "dq")
    # Same XYZ Euler -> wxyz quaternion -> inverse rotation as upstream.
    gravity = np.array([math.sin(pitch), -math.sin(roll)*math.cos(pitch),
                        -math.cos(roll)*math.cos(pitch)], dtype=np.float32)
    return np.concatenate((gyro, gravity, q-default, dq, np.zeros(22, np.float32)))


class ShadowInference:
    def __init__(self, model_dir=MODEL_DIR):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        contract_path = model_dir / "proprio_hold.contract.json"
        if hashlib.sha256(contract_path.read_bytes()).hexdigest() != EXPECTED_CONTRACT:
            raise ValueError("Contract hash differs from this reviewed model export")
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        c = self.contract
        expected = {"schema": "team15-proprio-hold-actor-v1", "actor_observation_dim": 72,
                    "action_dim": 22, "source_checkpoint_sha256": EXPECTED_CHECKPOINT,
                    "no_vision": True, "actor_uses_payload_state": False,
                    "normalization_included": True, "mass_observed": False,
                    "action_output_semantics": "clipped_normalized_joint_position_offset"}
        for key, value in expected.items():
            if c.get(key) != value:
                raise ValueError(f"Wrong model contract: {key}")
        model = model_dir / "proprio_hold.onnx"
        if hashlib.sha256(model.read_bytes()).hexdigest() != c["onnx_sha256"]:
            raise ValueError("Model hash differs from contract")
        self.default = vector(c["joint_mapping"]["default_joint_pos_rad"], 22, "default pose")
        self.scale = vector(c["joint_mapping"]["action_scale_rad"], 22, "action scale")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        ins, outs = self.session.get_inputs(), self.session.get_outputs()
        if len(ins) != 1 or ins[0].name != "obs" or ins[0].shape != [1,72] or ins[0].type != "tensor(float)":
            raise ValueError("Expected a single float32 [1,72] obs tensor")
        if len(outs) != 1 or outs[0].name != "actions" or outs[0].shape != [1,22]:
            raise ValueError("Expected a single [1,22] actions tensor")

    def infer(self, record):
        obs = observation(record, self.default)
        started = time.perf_counter()
        out = self.session.run(["actions"], {"obs": obs[None]})[0]
        elapsed = (time.perf_counter()-started)*1000
        if out.shape != (1,22) or not np.isfinite(out).all():
            raise ValueError("Invalid model output")
        if np.max(np.abs(out)) > 1.000001:
            raise ValueError("Exported action exceeded [-1,1]")
        actions = np.clip(out[0], -1, 1)
        return dict(observation=obs.tolist(), proposed_action=actions.tolist(),
                    proposed_unprojected_target_rad=(self.default+self.scale*actions).tolist(),
                    inference_ms=elapsed, commands_sent=0,
                    previous_action_mode="zero_each_frame_cold_start_only")


def run_live(args, emit, stats):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from booster_interface.msg import LowState
    # Intentionally no publishers, service clients, Portal or mode switches.
    rclpy.init(args=[])
    node = rclpy.create_node("team15_proprio_shadow_readonly")
    latest = {}

    def receive(message):
        try:
            latest.update(record=low_state_record(message), received=time.monotonic(),
                          sequence=stats["received_messages"]+1)
            stats["received_messages"] += 1
        except Exception as exc:
            latest["error"] = f"Invalid LowState: {exc}"

    start, last_sequence, last_emit = time.monotonic(), 0, -math.inf
    try:
        node.create_subscription(LowState, args.topic, receive,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                            history=HistoryPolicy.KEEP_LAST))
        while rclpy.ok() and time.monotonic()-start < args.seconds:
            rclpy.spin_once(node, timeout_sec=.01)
            now = time.monotonic()
            if "error" in latest:
                raise ValueError(latest["error"])
            if "record" not in latest:
                if now-start > args.startup_timeout:
                    raise TimeoutError("No LowState arrived; check ROS environment/domain/network")
                continue
            age = now-latest["received"]
            if age > args.max_age:
                raise TimeoutError(f"LowState reception stalled for {age:.3f}s; diagnostic stopped")
            if latest["sequence"] == last_sequence or now-last_emit < .02:
                continue
            emit(latest["record"], receive_age_s=age, elapsed_s=now-start,
                 sequence=latest["sequence"])
            if time.monotonic()-latest["received"] > args.max_age:
                raise TimeoutError("Inference/recording stalled beyond the state age limit")
            last_sequence, last_emit = latest["sequence"], now
        stats["actual_elapsed_s"] = time.monotonic()-start
        stats["completed_requested_duration"] = stats["actual_elapsed_s"] >= args.seconds
        if not stats["completed_requested_duration"]:
            raise RuntimeError("ROS stopped before the requested diagnostic duration completed")
    finally:
        stats["actual_elapsed_s"] = time.monotonic()-start
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-jsonl", type=Path, help="Replay recorded state fields without ROS; not a hardware test")
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--topic", default="/low_state")
    parser.add_argument("--startup-timeout", type=float, default=5.)
    parser.add_argument("--max-age", type=float, default=.1)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in [args.seconds,args.startup_timeout,args.max_age]):
        parser.error("Durations must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    stats = dict(schema="team15-readonly-shadow-v1", passed=False, commands_sent=0,
                 hardware_control_tested=False, source="replay" if args.replay_jsonl else "live_ros2",
                 previous_action_mode="zero_each_frame_cold_start_only", received_messages=0,
                 requested_duration_s=None if args.replay_jsonl else args.seconds,
                 completed_requested_duration=None if args.replay_jsonl else False,
                 samples=0, error=None,
                 limitations=["No motor command, mode RPC or real closed-loop control is performed.",
                              "Previous action is reset to zero every frame: diagnostic inputs, not an executed policy history.",
                              "Serial joint order and IMU conventions match pinned code, not verified physical calibration.",
                              "Freshness uses local receive time, not sensor timestamp; timing is not a hard real-time guarantee."])
    latencies = []
    try:
        engine = ShadowInference()
        stats["model_sha256"] = engine.contract["onnx_sha256"]
        stats["assumed_serial_joint_names"] = engine.contract["joint_mapping"]["booster_deploy_joint_names"]
        with (args.output/"samples.jsonl").open("w",encoding="utf-8") as handle:
            def emit(record, **extra):
                result = engine.infer(record)
                handle.write(json.dumps(dict(state=record, **extra, **result), allow_nan=False)+"\n")
                handle.flush()
                latencies.append(result["inference_ms"])
                stats["samples"] += 1
            if args.replay_jsonl:
                with args.replay_jsonl.open(encoding="utf-8") as source:
                    for line in source:
                        entry = json.loads(line)
                        emit(entry.get("state",entry))
                        stats["received_messages"] += 1
            else:
                run_live(args, emit, stats)
        if not stats["samples"]:
            raise ValueError("No diagnostic samples recorded")
        stats["passed"] = True
    except (Exception, KeyboardInterrupt) as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if latencies:
            stats["inference_ms_p50_p95_max"] = [float(v) for v in np.quantile(latencies,[.5,.95,1.])]
        (args.output/"summary.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(stats,indent=2))
    return 0 if stats["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
