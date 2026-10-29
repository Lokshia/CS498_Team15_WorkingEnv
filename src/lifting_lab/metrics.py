"""Strict episode-level evaluation; never replace max errors by averages."""
from __future__ import annotations

import numpy as np


def _finite_list(values):
    """Return strict-JSON statistics without hiding invalid values as zero."""
    return [float(value) if np.isfinite(value) else None for value in values]


def episode_success(steps, required_steps, fallen, max_tilt_deg, max_height_error_m,
                    tilt_limit_deg=10.0, height_limit_m=0.05):
    """All criteria must hold for a complete episode. Corrective steps are allowed."""
    return ((np.asarray(steps) >= required_steps) & ~np.asarray(fallen, dtype=bool)
            & np.isfinite(max_tilt_deg) & np.isfinite(max_height_error_m)
            & (np.asarray(max_tilt_deg) < tilt_limit_deg)
            & (np.asarray(max_height_error_m) < height_limit_m))


class EpisodeAccumulator:
    def __init__(self, num_envs, required_steps, tilt_limit_deg=10.0, height_limit_m=0.05,
                 *, support_mode="pregrasp", contact_grace_steps=6):
        if num_envs < 1 or required_steps < 1 or contact_grace_steps < 0:
            raise ValueError("Positive batch/episode length and nonnegative contact grace required")
        self.required_steps = required_steps
        self.support_mode = support_mode
        self.contact_grace_steps = contact_grace_steps
        self.tilt_limit_deg = tilt_limit_deg
        self.height_limit_m = height_limit_m
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.fallen = np.zeros(num_envs, dtype=bool)
        self.numerical_failure = np.zeros(num_envs, dtype=bool)
        self.early_truncation = np.zeros(num_envs, dtype=bool)
        self.disallowed_support = np.zeros(num_envs, dtype=bool)
        self.contact_failure = np.zeros(num_envs, dtype=bool)
        self.contact_gap = np.zeros(num_envs, dtype=np.int64)
        self.max_contact_gap = np.zeros(num_envs, dtype=np.int64)
        self.max_tilt = np.zeros(num_envs)
        self.max_height_error = np.zeros(num_envs)
        self.sums: dict[str, np.ndarray] = {}

    @property
    def active(self):
        return ~self.fallen & (self.steps < self.required_steps)

    def update(self, metrics, terminated, truncated=None):
        active = self.active.copy()
        for name in ("payload_tilt_deg", "height_error_m"):
            if np.asarray(metrics[name]).shape != self.steps.shape:
                raise ValueError(f"{name} must have one value per environment")
        tilt = np.asarray(metrics["payload_tilt_deg"])
        height = np.abs(np.asarray(metrics["height_error_m"]))
        self.steps[active] += 1
        self.max_tilt[active] = np.maximum(self.max_tilt[active], tilt[active])
        self.max_height_error[active] = np.maximum(self.max_height_error[active], height[active])
        finite = np.isfinite(tilt) & np.isfinite(height)
        for name, values in metrics.items():
            values = np.asarray(values)
            if values.shape == self.steps.shape:
                finite &= np.isfinite(values)
                self.sums.setdefault(name, np.zeros_like(values, dtype=float))
                with np.errstate(over="ignore", invalid="ignore"):
                    self.sums[name][active] += values[active]
                # Finite inputs can still overflow a statistic's running sum.
                finite &= np.isfinite(self.sums[name])
        self.numerical_failure[active] |= ~finite[active]
        if "nonfinite" in metrics:
            self.numerical_failure[active] |= np.asarray(metrics["nonfinite"], dtype=bool)[active]
        observed_fall = np.asarray(metrics.get("fell", np.zeros_like(active)), dtype=bool)
        if truncated is not None:
            self.early_truncation[active] |= (np.asarray(truncated, dtype=bool)[active]
                                              & (self.steps[active] < self.required_steps))
        if self.support_mode == "forearms":
            # Do not silently treat absent contact evidence as valid support.
            illegal = np.asarray(metrics["payload_disallowed_contact"], dtype=bool)
            lost = np.asarray(metrics["support_lost"], dtype=bool)
            supported = np.asarray(metrics["payload_forearm_supported"], dtype=bool)
            self.contact_gap[active & supported] = 0
            self.contact_gap[active & ~supported] += 1
            self.max_contact_gap[active] = np.maximum(self.max_contact_gap[active], self.contact_gap[active])
            self.disallowed_support[active] |= illegal[active]
            self.contact_failure[active] |= lost[active] | (self.contact_gap[active] > self.contact_grace_steps)
        self.fallen[active] |= (np.asarray(terminated, dtype=bool)[active]
                                | observed_fall[active]
                                | self.numerical_failure[active]
                                | self.early_truncation[active]
                                | self.disallowed_support[active] | self.contact_failure[active])

    def finish(self):
        success = episode_success(self.steps, self.required_steps, self.fallen,
                                  self.max_tilt, self.max_height_error,
                                  self.tilt_limit_deg, self.height_limit_m)
        means = {k: _finite_list(v / np.maximum(self.steps, 1)) for k, v in self.sums.items()}
        diagnostic_keys = (
            "joint_velocity_rms_rad_s", "action_delta_rms", "robot_vertical_speed_m_s",
            "robot_angular_speed_rad_s", "payload_linear_speed_m_s", "payload_angular_speed_rad_s",
            "both_feet_contact", "airborne", "payload_forearm_supported", "torque_rms_nm",
        )
        return {
            "success": success.tolist(), "success_rate": float(success.mean()),
            "steps": self.steps.tolist(), "fallen": self.fallen.tolist(),
            "numerical_failure": self.numerical_failure.tolist(),
            "early_truncation": self.early_truncation.tolist(),
            "disallowed_support": self.disallowed_support.tolist(),
            "contact_failure": self.contact_failure.tolist(),
            "max_contact_gap_steps": self.max_contact_gap.tolist(),
            "max_payload_tilt_deg": _finite_list(self.max_tilt),
            "max_abs_height_error_m": _finite_list(self.max_height_error),
            "means_until_completion_or_fall": means,
            "smoothness_summary": {
                "scope": "Diagnostic time-means until completion/failure; not a new success criterion. Short failed episodes are not comparable to full episodes without considering duration.",
                "per_episode_means": {key: means[key] for key in diagnostic_keys if key in means},
            },
        }

