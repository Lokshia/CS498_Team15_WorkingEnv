"""Read-only replay diagnostics; never a formal evaluation or policy-success test.

Usage: python scripts/analyze_hold_trace.py demo.json metadata/ --output NEW.json
Directories are searched for demo.json. Repeated paths are deduplicated. Only
stdlib and NumPy are used: no simulator, checkpoint loading, or GPU operations.
The sample interval is simulation_seconds / frames, not wall-clock time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "valid_hold", "payload_forearm_supported", "feet_contact_count",
    "support_gap_steps", "payload_tilt_deg", "height_error_m", "torque_rms_nm",
    "base_height_m", "payload_height_m", "fell", "nonfinite", "support_lost",
    "payload_disallowed_contact",
)
FLAG_FIELDS = (
    "valid_hold", "payload_forearm_supported", "fell", "nonfinite",
    "support_lost", "payload_disallowed_contact",
)
OPTIONAL_FIELDS = ("joint_velocity_rms_rad_s", "action_delta_rms")
OPTIONAL_PREFIXES = ("joint_velocity_", "joint_acceleration_", "action_", "torque_delta_")
MIN_SPECTRUM_SECONDS = 4.0
MIN_SPECTRUM_SAMPLES = 64


def _finite_number(value, label):
    if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite numeric scalar")
    return float(value)


def _check_finite_tree(value, label):
    """Also reject nonfinite values in unknown diagnostic fields/arrays."""
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{label}[{index}]")


def longest_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def statistics(values):
    with np.errstate(over="raise", invalid="raise"):
        return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)), "min": float(np.min(values)),
                "std": float(np.std(values))}


def spectrum(values, dt):
    """Linear detrend + periodic Hann + rFFT, without zero-padding.

    A spectral peak is descriptive, not a causal mechanism or proof of a stable
    oscillation. Near-record-length peaks can be drift. Frequencies above the
    sampling Nyquist limit, joint chatter, and aliasing cannot be resolved here.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    duration = n * dt
    result = {"peak_hz": None, "resolution_hz": 1.0 / duration,
              "nyquist_hz": 0.5 / dt, "possible_drift": None,
              "peak_three_bin_power_fraction": None, "status": "short_trace"}
    if n < MIN_SPECTRUM_SAMPLES or duration < MIN_SPECTRUM_SECONDS:
        result["note"] = "Frequency withheld: fewer than 64 samples or 4 seconds."
        return result
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        coordinate = np.arange(n, dtype=float) - (n - 1) / 2
        centered = values - values.mean()
        residual = centered - coordinate * (np.dot(coordinate, centered) / np.dot(coordinate, coordinate))
        tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
        if float(np.max(np.abs(residual))) <= tolerance:
            result.update(status="constant_or_linear_trend",
                          note="No resolved variation after removing the linear trend.")
            return result
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
        power = np.abs(np.fft.rfft(residual * window)) ** 2
        # One-sided periodogram weighting; Nyquist has no negative-frequency twin.
        power[1:-1 if n % 2 == 0 else None] *= 2
        power[0] = 0
        index = int(np.argmax(power))
        frequency = float(np.fft.rfftfreq(n, d=dt)[index])
        drift = frequency < 0.5 or frequency * duration < 3
        result.update(peak_hz=frequency, possible_drift=drift,
                      peak_three_bin_power_fraction=float(power[max(1, index - 1):index + 2].sum() / power.sum()),
                      status="possible_drift" if drift else "spectral_peak",
                      note=("Low-frequency/under-three-cycle peak may be drift, not an established oscillation."
                            if drift else "Sampled spectral peak only; not proof of its physical cause."))
    return result


def analyze_document(document):
    """Validate one demo.json object and summarize its original recorded rows."""
    if not isinstance(document, dict):
        raise ValueError("demo.json must contain an object")
    rows = document.get("trace")
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace must be a nonempty list")
    frames = document.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames != len(rows):
        raise ValueError("frames must be an integer equal to the trace row count")
    seconds = _finite_number(document.get("simulation_seconds"), "simulation_seconds")
    if isinstance(document.get("simulation_seconds"), bool) or seconds <= 0:
        raise ValueError("simulation_seconds must be positive")
    dt = seconds / frames
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(1.0 / dt):
        raise ValueError("Invalid inferred sample interval")
    columns = {key: [] for key in REQUIRED_FIELDS}
    all_keys = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"trace[{index}] must be an object")
        _check_finite_tree(row, f"trace[{index}]")
        all_keys.update(row)
        for key in REQUIRED_FIELDS:
            if key not in row:
                raise ValueError(f"trace[{index}] is missing required field {key}")
            columns[key].append(_finite_number(row[key], f"trace[{index}].{key}"))
    data = {key: np.asarray(values, dtype=float) for key, values in columns.items()}
    for key in FLAG_FIELDS:
        if not np.isin(data[key], (0, 1)).all():
            raise ValueError(f"{key} must contain only 0/1 flags")
    if not np.isin(data["feet_contact_count"], (0, 1, 2)).all():
        raise ValueError("feet_contact_count must be 0, 1, or 2")
    gap = data["support_gap_steps"]
    if (gap < 0).any() or not np.equal(gap, np.floor(gap)).all():
        raise ValueError("support_gap_steps must contain nonnegative integer counts")
    if (data["torque_rms_nm"] < 0).any() or (data["payload_tilt_deg"] < 0).any():
        raise ValueError("RMS torque and tilt magnitude must be nonnegative")

    optional = {}
    names = set(OPTIONAL_FIELDS) | {key for key in all_keys if key.startswith(OPTIONAL_PREFIXES)}
    for key in sorted(names):
        present = [key in row for row in rows]
        if not any(present):
            optional[key] = None
            continue
        if not all(present):
            raise ValueError(f"Optional diagnostic {key} is present in only part of the trace")
        values = np.asarray([_finite_number(row[key], f"trace[{i}].{key}") for i, row in enumerate(rows)])
        optional[key] = statistics(values)

    supported = data["payload_forearm_supported"] == 1
    valid = data["valid_hold"] == 1
    invalid = np.flatnonzero(~valid)
    both_arms_fraction = None
    if "forearm_contact_count" in all_keys:
        values = np.asarray([_finite_number(row.get("forearm_contact_count"),
                            f"trace[{i}].forearm_contact_count") for i, row in enumerate(rows)])
        if not np.isin(values, (0, 1, 2)).all():
            raise ValueError("forearm_contact_count must be 0, 1, or 2")
        both_arms_fraction = float((values == 2).mean())
    result = {
        "formal_evaluation": False,
        "frames": frames, "simulation_seconds": seconds, "dt_seconds": dt,
        "sampling_rate_hz": 1.0 / dt,
        "valid_hold": {"all_recorded_steps_valid": bool(valid.all()),
                       "valid_step_fraction": float(valid.mean()),
                       "invalid_step_count": int(len(invalid)),
                       "first_invalid_step_1_based": int(invalid[0] + 1) if len(invalid) else None},
        "contact": {"bilateral_forearm_support_fraction": float(supported.mean()),
                    "both_forearms_contact_fraction": both_arms_fraction,
                    "both_feet_contact_fraction": float((data["feet_contact_count"] == 2).mean()),
                    "neither_foot_contact_fraction": float((data["feet_contact_count"] == 0).mean()),
                    "max_support_gap_steps": int(gap.max()),
                    "max_support_gap_seconds": float(gap.max() * dt),
                    "longest_recorded_unsupported_run_steps": longest_run(~supported),
                    "longest_no_feet_contact_steps": longest_run(data["feet_contact_count"] == 0)},
        "payload_tilt_deg": {"max": float(data["payload_tilt_deg"].max()),
                             "p95": float(np.percentile(data["payload_tilt_deg"], 95))},
        "height_error_absmax_m": float(np.abs(data["height_error_m"]).max()),
        "torque_rms_nm": statistics(data["torque_rms_nm"]),
        "failure_flag_counts": {key: int((data[key] == 1).sum())
                                for key in ("fell", "nonfinite", "support_lost", "payload_disallowed_contact")},
        "optional_diagnostics": optional,
        "provenance": {"support_mode": document.get("support_mode"), "seed": document.get("seed"),
                       "checkpoint_sha256": document.get("checkpoint_sha256"),
                       "runtime_source_sha256": document.get("runtime", {}).get("source_sha256")},
    }
    for key in ("base_height_m", "payload_height_m"):
        result[key] = {"std": float(data[key].std()), "peak_to_peak": float(np.ptp(data[key])),
                       "spectrum": spectrum(data[key], dt)}
    # Prevent overflowed aggregate values from ever becoming non-standard JSON.
    _check_finite_tree(result, "summary")
    return result


def collect_paths(inputs):
    found = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            matches = [p for p in path.rglob("demo.json") if p.is_file()]
            if not matches:
                raise ValueError(f"No demo.json files found in {path}")
            found.update(p.resolve() for p in matches)
        else:
            raise ValueError(f"Input does not exist: {path}")
    if not found:
        raise ValueError("No input traces")
    return sorted(found, key=lambda p: str(p).casefold())


def analyze_paths(inputs):
    results = []
    for path in collect_paths(inputs):
        raw = path.read_bytes()
        try:
            result = analyze_document(json.loads(raw))
        except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        results.append({"name": path.parent.name, "source": str(path),
                        "source_file_sha256": hashlib.sha256(raw).hexdigest(), **result})
    return {"schema_version": 1, "formal_evaluation": False,
            "scope": "Replay diagnostics only; no formal episode-success or learned-policy claim.",
            "trace_count": len(results),
            "frequency_method": {"detrend": "linear", "window": "periodic Hann", "transform": "one-sided rFFT",
                                 "minimum_seconds": MIN_SPECTRUM_SECONDS, "minimum_samples": MIN_SPECTRUM_SAMPLES,
                                 "possible_drift_rule": "peak < 0.5 Hz OR fewer than three cycles in the recording"},
            "limitations": ["Uniform sampling is inferred from simulation_seconds/frames; no per-row timestamps were checked.",
                            "valid_hold summarizes recorded flags; it is not the formal multi-episode protocol.",
                            "Contact fractions are sampled sensor flags, not continuous-time proof of support or flight.",
                            "Spectral peaks do not identify policy/PD/contact causes or frequencies above Nyquist.",
                            "Absent joint/action diagnostics remain null; RMS torque cannot recover individual-joint peaks or energy."],
            "traces": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="demo.json files or directories searched recursively")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    summary = analyze_paths(args.inputs)
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another writer after validation.
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    print(json.dumps({"diagnostic_traces": summary["trace_count"],
                      "formal_evaluation": False, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
