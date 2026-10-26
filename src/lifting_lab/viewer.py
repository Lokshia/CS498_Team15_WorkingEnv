"""Interactive mass changes, applied by rebuilding physics on the main thread.

No in-place mass edits: MuJoCo and Warp derived inertial quantities must be
rebuilt together. The browser/server stays connected while each world is replaced.
"""
from __future__ import annotations

from copy import copy
import math
from threading import RLock

import viser
from booster_mjlab.viewer import RecordingViserPlayViewer
from mjlab.rl import RslRlVecEnvWrapper

from .runtime import file_sha256, runtime_info, write_json

MAX_VIEWER_MASS_KG = 10.0  # Exploratory UI range, not a robot payload rating.


def validate_viewer_mass(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= MAX_VIEWER_MASS_KG:
        raise ValueError(f"Weight must be finite and between 0 and {MAX_VIEWER_MASS_KG:g} kg.")
    return value


class MassAdjustableViewer(RecordingViserPlayViewer):
    def __init__(self, *args, mass, notice="", support_mode="pregrasp", **kwargs):
        super().__init__(*args, **kwargs)
        self.mass = mass
        self.notice = notice
        self.support_mode = support_mode
        self.requested_mass_kg = None
        self._pending_mass = None
        self._lifecycle_lock = RLock()
        self._closing = False
        self._closed = False
        self._record_operations = 0
        self._owned_connect_callbacks = []
        self._owned_disconnect_callbacks = []

    def setup(self):
        # These private hooks are pinned to Viser 1.1.0. Its public GUI reset
        # does not unregister mjviser's camera-connect callback. Track only our
        # own additions so unrelated server callbacks remain intact.
        with self._server._client_lock:
            old_connect = tuple(self._server._client_connect_cb)
            old_disconnect = tuple(self._server._client_disconnect_cb)
        if getattr(self, "support_mode", "pregrasp") == "forearms":
            support_description = "自由箱体：仅靠前臂/手部接触承托，无 CONNECT/WELD 或粘附；可能滑落。预览不因跌倒终止或自动重置，可手动 Reset。"
        else:
            support_description = "旧预抓握模式：两处 CONNECT 点连接固定负载，不是真实摩擦抓握。"
        with self._server.gui.add_folder("Payload / 负重"):
            self._weight_input = self._server.gui.add_number(
                "重量 (kg)", initial_value=self.mass["requested_payload_mass_kg"],
                min=0.0, max=MAX_VIEWER_MASS_KG, step=0.05,
                hint="Only changes after Apply & Reset. 0 kg uses a 1 g numerical floor.")
            self._apply_mass_button = self._server.gui.add_button("应用并重置 / Apply & Reset")
            self._mass_status = self._server.gui.add_markdown(
                f"**当前实际质量：{self.mass['actual_payload_mass_kg']:.3f} kg**\n\n"
                f"{support_description}\n\n"
                "未加载策略时为零动作关节位置 PD，不是训练好的平衡策略；Play 后可能跌倒。\n\n"
                "应用后重建质量和惯量，回到初始姿态并暂停。\n\n"
                "0 kg 实际使用 0.001 kg；0–10 kg 仅为仿真范围，不代表真机承载能力。")
            self._mass_notice = self._server.gui.add_markdown(self.notice or "等待修改。")

        @self._apply_mass_button.on_click
        def _(_event):
            self.request_mass(self._weight_input.value)

        try:
            super().setup()
            # Dynamic mesh instances start at the origin. Upstream only syncs
            # a paused scene when flagged dirty, so publish the initial FK now
            # without stepping physics (also exposes sync errors immediately).
            with self._sim_lock, self._server.atomic():
                self._scene.paused = self._is_paused
                self._scene.update(self.env.unwrapped.sim.data)
                self._server.flush()
        finally:
            with self._server._client_lock:
                self._owned_connect_callbacks = [cb for cb in self._server._client_connect_cb
                                                 if cb not in old_connect]
                self._owned_disconnect_callbacks = [cb for cb in self._server._client_disconnect_cb
                                                    if cb not in old_disconnect]

    def request_mass(self, value):
        """Thread-safe UI request. Never touches physics or closes an environment."""
        with self._lifecycle_lock:
            if self._closing or self._pending_mass is not None:
                return False
            if self._is_recording or self._record_operations:
                self._mass_notice.content = "请先结束录像，再修改重量。"
                return False
            try:
                self._pending_mass = validate_viewer_mass(value)
            except (ValueError, TypeError, OverflowError):
                self._mass_notice.content = "重量必须是 0–10 kg 的有限数值。"
                return False
            self._weight_input.disabled = True
            self._apply_mass_button.disabled = True
            self._record_button.disabled = True
            self._mass_notice.content = "正在重建物理模型，请稍候……"
            return True

    def tick(self):
        with self._lifecycle_lock:
            if self._pending_mass is not None:
                # UI recording starts are blocked as soon as the request is
                # accepted. Queueing Play/Step cannot advance the retiring world.
                self.pause()
                self.requested_mass_kg = self._pending_mass
                self._closing = True
                self._actions.clear()
                self._interrupted = True
                return False
        return super().tick()

    def _start_recording(self, client):
        with self._lifecycle_lock:
            if (self._closing or self._pending_mass is not None
                    or self._record_operations or self._is_recording):
                return
            return super()._start_recording(client)

    def _stop_recording(self, *args, **kwargs):
        # Do not hold the lock while the recorder joins its worker thread.
        with self._lifecycle_lock:
            self._record_operations += 1
        try:
            return super()._stop_recording(*args, **kwargs)
        finally:
            with self._lifecycle_lock:
                self._record_operations -= 1

    def _finish_lockstep(self, *args, **kwargs):
        with self._lifecycle_lock:
            self._record_operations += 1
        try:
            return super()._finish_lockstep(*args, **kwargs)
        finally:
            with self._lifecycle_lock:
                self._record_operations -= 1

    def close(self):
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._closing = True
        try:
            # setup() may fail before Viser allocates its scene worker.
            if hasattr(self, "_threadpool"):
                super().close()
        finally:
            with self._server._client_lock:
                self._server._client_connect_cb[:] = [
                    cb for cb in self._server._client_connect_cb if cb not in self._owned_connect_callbacks]
                self._server._client_disconnect_cb[:] = [
                    cb for cb in self._server._client_disconnect_cb if cb not in self._owned_disconnect_callbacks]

    def run(self, *args, **kwargs):
        # Upstream run calls setup outside its try/finally. Cover partial setup
        # failures too; close is idempotent for the normal upstream close path.
        try:
            return super().run(*args, **kwargs)
        finally:
            self.close()


def prepare_world(args, env_factory, policy_factory):
    """Build first, seed after every wrapper reset, and close on partial failure."""
    from .cli import _validate_checkpoint_support
    _validate_checkpoint_support(args)
    env, mass = env_factory(args, seconds=args.seconds + 1.0,
                           auto_reset=getattr(args, "support_mode", "pregrasp") == "pregrasp",
                           num_envs=1)
    try:
        policy = policy_factory(env, args)
        wrapper = RslRlVecEnvWrapper(env)
        env.reset(seed=args.seed)
        return env, mass, policy, wrapper
    except BaseException:
        env.close()
        raise


def world_snapshot(env, mass, revision, paused):
    body_id = env.sim.mj_model.body("payload/box").id
    return {"revision": revision, "mass": mass, "paused": paused,
            "support_mode": getattr(env.cfg, "support_mode", "pregrasp"),
            "equality_constraints": int(env.sim.mj_model.neq),
            "compiled_payload_mass_kg": float(env.sim.mj_model.body_mass[body_id]),
            "compiled_payload_inertia_kg_m2": env.sim.mj_model.body_inertia[body_id].tolist(),
            "warp_payload_mass_kg": float(env.sim.model.body_mass[..., body_id].mean()),
            "simulation_time_s": float(env.sim.data.time[0]),
            "control_steps": int(env.episode_length_buf[0])}


def run_interactive(args, output, env_factory, policy_factory):
    env, mass, policy, wrapper = prepare_world(args, env_factory, policy_factory)
    server = None
    history = []
    local = copy(args)
    paused = args.paused
    notice = ""
    try:
        validate_viewer_mass(mass["requested_payload_mass_kg"])
        server = viser.ViserServer(host="127.0.0.1", label="K1 Lifting Lab")
        while True:
            snapshot = world_snapshot(env, mass, len(history), paused)
            history.append(snapshot)
            metadata = {"runtime": runtime_info(env.device), "mass": mass,
                        "support_mode": snapshot["support_mode"],
                        "seed": args.seed, "mass_observed": not args.mass_blind,
                        "policy": "checkpoint" if args.checkpoint else "zero-action joint-position PD; no trained balancing policy",
                        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
                        "mode": "interactive viewer", "start_paused": paused,
                        "requested_simulation_seconds": args.seconds, "recordings": "recordings/",
                        "mass_change_history": history,
                        "note": "Each applied mass rebuilds physics and resets paused. Viewer is not an evaluation."}
            write_json(output / "demo.json", metadata)
            write_json(output / f"mass_revision_{snapshot['revision']:03d}.json", snapshot)
            viewer = MassAdjustableViewer(wrapper, policy=lambda td: policy(dict(td)),
                       mass=mass, notice=notice, support_mode=snapshot["support_mode"],
                       video_dir=output / "recordings", viser_server=server)
            if paused:
                viewer.pause()
                print(f"Viewer PAUSED: requested={mass['requested_payload_mass_kg']:.3f} kg, "
                      f"actual={mass['actual_payload_mass_kg']:.3f} kg, time={snapshot['simulation_time_s']:.3f}s", flush=True)
            viewer.run(num_steps=round(args.seconds / env.step_dt))
            if viewer.requested_mass_kg is None:
                break
            updated = copy(local)
            updated.mass = viewer.requested_mass_kg
            updated.mass_fraction = None
            updated.mass_range_fraction = None
            try:
                new_env, new_mass, new_policy, new_wrapper = prepare_world(updated, env_factory, policy_factory)
            except Exception as exc:
                # Old physics is still intact; return to its initial pose.
                env.reset(seed=args.seed)
                notice = f"修改失败，仍使用 {mass['actual_payload_mass_kg']:.3f} kg。{type(exc).__name__}: {exc}"
                print(notice, flush=True)
            else:
                env.close()
                env, mass, policy, wrapper = new_env, new_mass, new_policy, new_wrapper
                local = updated
                notice = "已应用。当前暂停；Play 连续运行，Step 单步。"
            paused = True
            # Old viewer/recording/scene threads and its camera callbacks are
            # gone before deleting their handles. Keep this same HTTP server.
            server.gui.reset()
            server.scene.reset()
    finally:
        if server is not None:
            server.stop()
        env.close()
