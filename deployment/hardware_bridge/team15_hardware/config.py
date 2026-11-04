"""Fail-closed site configuration validation, with no ROS or network imports.

A syntactically valid profile records site-specific assertions; it does not
prove those assertions. Simulation fixtures are explicitly marked and must
never be accepted by a live launcher.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


SCHEMA = "team15-hardware-site-v1"
N = 22
DT = 0.02
UPSTREAM_COMMIT = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"
MODEL_SHA = "c75c00f943ddca0a94ed6722a9848a2443909150aaffc63d253f184b4b55b921"
PROTOCOL = {
    "change_mode_api": 2000,
    "get_status_api": 2018,
    "mode_values": {"damping": 0, "walking": 2, "custom": 3},
    "command_type": "serial",
}
CONFIRMATIONS = (
    "joint_interface_is_joint_level", "calibration_verified",
    "firmware_watchdog_verified", "exclusive_command_authority",
    "entry_exit_reviewed", "physical_support_and_estop_ready",
)
JOINT_POSITIVE_FIELDS = (
    "kp", "kd", "target_rate_limit_rad_s", "measured_speed_limit_rad_s",
    "measured_torque_limit_nm", "tracking_error_limit_rad",
)
LIMIT_FIELDS = (
    "state_timeout_s", "command_timeout_s", "inference_timeout_s",
    "mode_timeout_s", "prime_settle_s", "prepare_timeout_s",
    "ready_timeout_s", "run_timeout_s", "max_trunk_tilt_deg",
    "subscriber_timeout_s", "mode_check_interval_s", "status_rpc_timeout_s",
)


class ConfigurationError(ValueError):
    def __init__(self, errors):
        self.errors = tuple(errors) if not isinstance(errors, str) else (errors,)
        super().__init__("Invalid site configuration:\n" + "\n".join("- " + e for e in self.errors))


class _Check:
    def __init__(self):
        self.errors = []

    def get(self, mapping, key, path):
        value = mapping.get(key) if isinstance(mapping, dict) else None
        if value is None:
            self.errors.append(f"{path}: missing or null; site value is required")
        return value

    def obj(self, value, path):
        if not isinstance(value, dict):
            if value is not None:
                self.errors.append(f"{path}: expected object")
            return {}
        return value

    def number(self, value, path, positive=False):
        if value is None:
            return None
        try:
            valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
        except (OverflowError, ValueError, TypeError):
            valid = False
        if not valid:
            self.errors.append(f"{path}: expected finite number")
            return None
        value = float(value)
        if positive and value <= 0:
            self.errors.append(f"{path}: must be positive")
            return None
        return value

    def text(self, value, path):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            self.errors.append(f"{path}: expected nonempty string")
            return None
        if value.strip().lower() in {"todo", "tbd", "unknown", "replace_me", "null"}:
            self.errors.append(f"{path}: placeholder is not a reviewed site value")
            return None
        return value

    def vector(self, value, path, length=N):
        if value is None:
            self.errors.append(f"{path}: missing or null; site value is required")
            return None
        if not isinstance(value, list) or len(value) != length:
            self.errors.append(f"{path}: expected list of exactly {length} numbers")
            return None
        values = []
        for i, item in enumerate(value):
            if item is None:
                self.errors.append(f"{path}[{i}]: missing or null; site value is required")
                values.append(None)
            else:
                values.append(self.number(item, f"{path}[{i}]"))
        return None if any(v is None for v in values) else np.array(values, dtype=float)

    def exact(self, actual, expected, path):
        if actual is not None and (type(actual) is not type(expected) or actual != expected):
            self.errors.append(f"{path}: expected {expected!r}")


def validate_config(data: dict, contract: dict) -> dict:
    """Return a defensive copy or report *all* absent/null/invalid site fields."""
    c = _Check()
    d = c.obj(data, "site")
    if not isinstance(data, dict):
        c.errors.append("site: expected JSON object")
    c.exact(c.get(d, "schema", "schema"), SCHEMA, "schema")
    deployment_target = c.get(d, "deployment_target", "deployment_target")
    if deployment_target is not None and deployment_target not in ("real_robot", "simulation_only"):
        c.errors.append("deployment_target: expected real_robot or simulation_only")
    c.exact(c.get(d, "upstream_booster_deploy_commit", "upstream_booster_deploy_commit"),
            UPSTREAM_COMMIT, "upstream_booster_deploy_commit")
    model = c.obj(c.get(d, "model", "model"), "model")
    for key, expected in (("onnx_sha256", MODEL_SHA), ("observation_dim", 72), ("action_dim", N)):
        c.exact(c.get(model, key, f"model.{key}"), expected, f"model.{key}")
    model_dt = c.number(c.get(model, "control_dt_s", "model.control_dt_s"), "model.control_dt_s")
    if model_dt is not None and model_dt != DT:
        c.errors.append("model.control_dt_s: expected exactly 0.02 seconds")

    review = c.obj(c.get(d, "review", "review"), "review")
    for key in ("robot_serial", "reviewer", "evidence"):
        c.text(c.get(review, key, f"review.{key}"), f"review.{key}")
    for key in CONFIRMATIONS:
        value = c.get(review, key, f"review.{key}")
        if value is not None and value is not True:
            c.errors.append(f"review.{key}: must be true after actual site review")

    protocol = c.obj(c.get(d, "protocol", "protocol"), "protocol")
    for key in ("change_mode_api", "get_status_api", "command_type"):
        c.exact(c.get(protocol, key, f"protocol.{key}"), PROTOCOL[key], f"protocol.{key}")
    modes = c.obj(c.get(protocol, "mode_values", "protocol.mode_values"), "protocol.mode_values")
    for key, expected in PROTOCOL["mode_values"].items():
        c.exact(c.get(modes, key, f"protocol.mode_values.{key}"), expected, f"protocol.mode_values.{key}")

    transport = c.obj(c.get(d, "transport", "transport"), "transport")
    for key in ("low_state_topic", "joint_ctrl_topic", "rpc_service"):
        text = c.text(c.get(transport, key, f"transport.{key}"), f"transport.{key}")
        if text is not None and (not text.startswith("/") or any(ch.isspace() for ch in text)):
            c.errors.append(f"transport.{key}: expected absolute ROS name without whitespace")
    rpc_timeout = c.number(c.get(transport, "rpc_timeout_s", "transport.rpc_timeout_s"),
                           "transport.rpc_timeout_s", positive=True)
    weights = c.vector(c.get(transport, "motor_weight", "transport.motor_weight"), "transport.motor_weight")
    if weights is not None and (np.any(weights < 0) or np.any(weights > 1)):
        c.errors.append("transport.motor_weight: all reviewed values must be between 0 and 1")

    imu = c.obj(c.get(d, "imu", "imu"), "imu")
    c.exact(c.get(imu, "convention", "imu.convention"), "rpy_xyz_imu_to_world_rad", "imu.convention")
    c.exact(c.get(imu, "gyro_frame", "imu.gyro_frame"), "imu", "imu.gyro_frame")
    rotation = c.get(imu, "body_from_imu", "imu.body_from_imu")
    if rotation is not None:
        if not isinstance(rotation, list) or len(rotation) != 3:
            c.errors.append("imu.body_from_imu: expected 3x3 rotation matrix")
        else:
            rows = [c.vector(row, f"imu.body_from_imu[{i}]", 3) for i, row in enumerate(rotation)]
            if all(row is not None for row in rows):
                matrix = np.array(rows)
                if (not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-7, rtol=0)
                        or not np.isclose(np.linalg.det(matrix), 1., atol=1e-7, rtol=0)):
                    c.errors.append("imu.body_from_imu: must be a proper SO(3) rotation (orthonormal, determinant +1)")

    try:
        mapping = contract["joint_mapping"]
        names = mapping["booster_deploy_joint_names"]
        nominal = np.asarray(mapping["default_joint_pos_rad"], dtype=float)
        scale = np.asarray(mapping["action_scale_rad"], dtype=float)
        if len(names) != N or len(set(names)) != N or nominal.shape != (N,) or scale.shape != (N,):
            raise ValueError
        if not np.isfinite(nominal).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ConfigurationError(["contract: invalid canonical 22-joint mapping"])
    joint_rows = c.get(d, "joints", "joints")
    if not isinstance(joint_rows, list) or len(joint_rows) != N:
        c.errors.append("joints: expected exactly 22 rows in canonical actor order")
        joint_rows = [None] * N
    lower, upper, rates, indices = [], [], [], []
    for i, row in enumerate(joint_rows):
        path = f"joints[{i}]"
        row = c.obj(row, path)
        c.exact(c.get(row, "name", path + ".name"), names[i], path + ".name")
        index = c.get(row, "serial_index", path + ".serial_index")
        if index is not None and (type(index) is not int or not 0 <= index < N):
            c.errors.append(path + ".serial_index: expected integer 0..21")
            index = None
        indices.append(index)
        sign = c.number(c.get(row, "sign", path + ".sign"), path + ".sign")
        if sign is not None and sign not in (-1., 1.):
            c.errors.append(path + ".sign: expected -1 or +1")
        c.number(c.get(row, "zero_offset_rad", path + ".zero_offset_rad"), path + ".zero_offset_rad")
        lo = c.number(c.get(row, "model_min_rad", path + ".model_min_rad"), path + ".model_min_rad")
        hi = c.number(c.get(row, "model_max_rad", path + ".model_max_rad"), path + ".model_max_rad")
        lower.append(lo); upper.append(hi)
        if lo is not None and hi is not None:
            if lo >= hi:
                c.errors.append(path + ": model_min_rad must be below model_max_rad")
            if not lo <= nominal[i] <= hi:
                c.errors.append(path + ": model nominal pose is outside reviewed joint range")
        positive = {}
        for key in JOINT_POSITIVE_FIELDS:
            positive[key] = c.number(c.get(row, key, path + "." + key), path + "." + key, positive=True)
        rates.append(positive["target_rate_limit_rad_s"])
    if all(i is not None for i in indices) and sorted(indices) != list(range(N)):
        c.errors.append("joints.serial_index: indices must form a unique permutation of 0..21")

    limits = c.obj(c.get(d, "limits", "limits"), "limits")
    lim = {key: c.number(c.get(limits, key, f"limits.{key}"), f"limits.{key}", positive=True)
           for key in LIMIT_FIELDS}
    for key in ("state_timeout_s", "command_timeout_s"):
        if lim[key] is not None and lim[key] > .2:
            c.errors.append(f"limits.{key}: must be at most 0.2 seconds")
    if lim["command_timeout_s"] is not None and lim["command_timeout_s"] < DT:
        c.errors.append("limits.command_timeout_s: must be at least the 0.02-second policy period")
    if lim["inference_timeout_s"] is not None and lim["inference_timeout_s"] >= DT:
        c.errors.append("limits.inference_timeout_s: must be below the 0.02-second policy period")
    if lim["max_trunk_tilt_deg"] is not None and lim["max_trunk_tilt_deg"] >= 45:
        c.errors.append("limits.max_trunk_tilt_deg: must be below 45 degrees")
    if rpc_timeout is not None and lim["mode_timeout_s"] is not None and rpc_timeout > lim["mode_timeout_s"]:
        c.errors.append("transport.rpc_timeout_s: must not exceed limits.mode_timeout_s")
    if (lim["status_rpc_timeout_s"] is not None and lim["mode_timeout_s"] is not None
            and lim["status_rpc_timeout_s"] > lim["mode_timeout_s"]):
        c.errors.append("limits.status_rpc_timeout_s: must not exceed limits.mode_timeout_s")

    prepare = c.obj(c.get(d, "prepare", "prepare"), "prepare")
    start = c.vector(c.get(prepare, "start_model_q", "prepare.start_model_q"), "prepare.start_model_q")
    c.number(c.get(prepare, "start_tolerance_rad", "prepare.start_tolerance_rad"),
             "prepare.start_tolerance_rad", positive=True)
    for key in ("ready_tolerance_rad", "ready_dwell_s", "ready_speed_limit_rad_s"):
        c.number(c.get(prepare, key, "prepare." + key), "prepare." + key, positive=True)
    waypoints = c.get(prepare, "waypoints", "prepare.waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        c.errors.append("prepare.waypoints: expected a nonempty list of reviewed waypoints")
        waypoints = []
    ranges_valid = all(v is not None for v in lower + upper)
    rates_valid = all(v is not None for v in rates)
    previous, elapsed, final = start, 0., None
    if start is not None and ranges_valid and (np.any(start < lower) or np.any(start > upper)):
        c.errors.append("prepare.start_model_q: starting pose is outside reviewed joint range")
    for i, item in enumerate(waypoints):
        path = f"prepare.waypoints[{i}]"
        item = c.obj(item, path)
        target = c.vector(c.get(item, "model_q", path + ".model_q"), path + ".model_q")
        duration = c.number(c.get(item, "duration_s", path + ".duration_s"), path + ".duration_s", positive=True)
        if target is not None:
            if ranges_valid and (np.any(target < lower) or np.any(target > upper)):
                c.errors.append(path + ".model_q: waypoint outside reviewed joint range")
            if previous is not None and duration is not None and rates_valid:
                if np.any(1.5 * np.abs(target - previous) / duration > np.asarray(rates) + 1e-12):
                    c.errors.append(path + ": smoothstep peak target speed exceeds reviewed per-joint rate limit")
        previous, final = target, target
        if duration is not None:
            elapsed += duration
    if final is not None and not np.array_equal(final, nominal):
        c.errors.append("prepare.waypoints[-1].model_q: final pose must exactly equal the model nominal pose")
    if lim["prepare_timeout_s"] is not None and elapsed > lim["prepare_timeout_s"]:
        c.errors.append("prepare.waypoints: total duration exceeds limits.prepare_timeout_s")

    stop = c.obj(c.get(d, "stop", "stop"), "stop")
    exit_mode = c.get(stop, "exit_mode", "stop.exit_mode")
    if exit_mode is not None and exit_mode not in ("damping", "walking"):
        c.errors.append("stop.exit_mode: expected damping or walking after site review")
    c.number(c.get(stop, "firmware_watchdog_timeout_s", "stop.firmware_watchdog_timeout_s"),
             "stop.firmware_watchdog_timeout_s", positive=True)
    c.text(c.get(stop, "procedure", "stop.procedure"), "stop.procedure")

    if c.errors:
        raise ConfigurationError(list(dict.fromkeys(c.errors)))
    return deepcopy(data)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError([f"JSON: duplicate key {key!r}"])
        result[key] = value
    return result


def load_config(path, contract: dict) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError([f"site file: {type(exc).__name__}: {exc}"]) from exc
    return validate_config(data, contract)
