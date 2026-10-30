"""One viewer for separately trained empty-hand and forearm-box policies.

Applying a selection rebuilds and resets the world; this is not continuous
loading/unloading or a hardware policy-switching controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import viser
from mjlab.rl import RslRlVecEnvWrapper

from . import cli, empty_hand_cli
from .runtime import file_sha256, runtime_info, write_json
from .viewer import MassAdjustableViewer, validate_viewer_mass, world_snapshot

MODES = {"empty": "空手 / No box", "box": "托箱 / With box"}
DEFAULT_BOX_MASS = 0.4949224923474894


@dataclass(frozen=True)
class Selection:
    mode: str
    mass_kg: float = 0.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("Unknown mode")
        if self.mode == "empty":
            object.__setattr__(self, "mass_kg", 0.0)
        else:
            mass = validate_viewer_mass(self.mass_kg)
            if mass < 0.001:
                raise ValueError("托箱重量须为 0.001–10 kg；没有物体请选择空手。")
            object.__setattr__(self, "mass_kg", mass)


def hold_args(args, selection):
    return cli.build_parser().parse_args([
        "demo", "--device", args.device, "--num-envs", "1", "--seed", str(args.seed),
        "--seconds", str(args.seconds), "--threads", "4", "--viewer", "--paused",
        "--support-mode", "forearms", "--mass", str(selection.mass_kg),
        "--checkpoint", str(args.box_checkpoint),
    ])


@dataclass
class World:
    env: object
    wrapper: object
    policy: object
    selection: Selection
    mass: dict
    checkpoint: Path

    def close(self):
        self.env.close()

    def snapshot(self, revision):
        env = self.env
        obs = self.wrapper.get_observations()
        result = {
            "revision": revision, "mode": self.selection.mode,
            "requested_mass_kg": self.selection.mass_kg,
            "payload_entity_present": "payload" in env.scene.entities,
            "paused": True, "simulation_time_s": float(env.sim.data.time[0]),
            "control_steps": int(env.episode_length_buf[0]),
            "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
            "equality_constraints": int(env.sim.mj_model.neq),
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": file_sha256(self.checkpoint),
        }
        if self.selection.mode == "box":
            result["payload_physics"] = world_snapshot(env, self.mass, revision, True)
        return result


def prepare_selection(args, selection):
    """No GUI or server. Close partially prepared worlds on any failure."""
    if selection.mode == "empty":
        empty_hand_cli.read_metadata(args.empty_checkpoint)
        env_args = SimpleNamespace(command="demo", device=args.device, num_envs=1,
            seed=args.seed, seconds=args.seconds, joint_jitter=.015, reset_speed=.02, push_speed=0)
        env = empty_hand_cli.environment(env_args)
        checkpoint = args.empty_checkpoint
        mass = {"requested_payload_mass_kg": 0.0, "actual_payload_mass_kg": 0.0}
    else:
        env_args = hold_args(args, selection)
        env, mass = cli._env(env_args, seconds=args.seconds+1, auto_reset=False, num_envs=1)
        checkpoint = args.box_checkpoint
    try:
        actor = (empty_hand_cli.policy(env, checkpoint) if selection.mode == "empty"
                 else cli._policy(env, env_args))
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs, _ = env.reset(seed=args.seed)
        dimensions = (72, 76) if selection.mode == "empty" else (97, 97)
        assert tuple(obs["actor"].shape) == (1, dimensions[0])
        assert tuple(obs["critic"].shape) == (1, dimensions[1])
        assert ("payload" in env.scene.entities) == (selection.mode == "box")
        with torch.inference_mode():
            action = actor(obs)
            assert tuple(action.shape) == (1, 22) and torch.isfinite(action).all()
        return World(env, wrapper, actor, selection, mass, checkpoint)
    except BaseException:
        env.close()
        raise


class ModeViewer(MassAdjustableViewer):
    def __init__(self, *args, selection, box_mass, training_range, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = selection
        self.box_mass = box_mass
        self.training_range = training_range
        self.requested_selection = None

    def _add_payload_controls(self):
        with self._server.gui.add_folder("场景与重量 / Load controls"):
            self._mode_input = self._server.gui.add_dropdown(
                "模式 / Mode", options=tuple(MODES.values()), initial_value=MODES[self.selection.mode])
            self._weight_input = self._server.gui.add_number(
                "箱子重量 (kg)", initial_value=self.box_mass, min=.001, max=10., step=.05,
                disabled=self.selection.mode == "empty",
                hint="选择托箱后输入。应用会重置场景并暂停。无箱子请选择空手。")
            self._apply_mass_button = self._server.gui.add_button("应用并重置 / Apply & Reset")
            if self.selection.mode == "empty":
                current = "**当前：空手，没有箱子实体。**\n\n已加载空手稳站策略。"
            else:
                current = (f"**当前：托箱，实际质量 {self.selection.mass_kg:.3f} kg。**\n\n"
                           "已加载托箱策略；箱子自由接触前臂，没有粘住。")
            low, high = self.training_range
            outside = self.selection.mode == "box" and not low <= self.selection.mass_kg <= high
            range_note = ("**当前超出托箱模型训练范围，可能站不稳。**" if outside else
                          "训练范围内也不保证每次成功。")
            self._mass_status = self._server.gui.add_markdown(
                f"{current}\n\n托箱训练范围：{low:.3f}–{high:.3f} kg。{range_note}\n\n"
                "修改后点应用，再点 Play。每次从初始姿态重新开始，不模拟运行中接箱或卸箱。"
                "\n\n0.001–10 kg 只是仿真输入范围，不是真机额定承重。")
            self._mass_notice = self._server.gui.add_markdown(self.notice or "当前暂停，点击 Play 播放。")

        @self._mode_input.on_update
        def _(_event):
            with self._lifecycle_lock:
                if not self._closing and self._pending_mass is None:
                    self._weight_input.disabled = self._mode_input.value == MODES["empty"]

        @self._apply_mass_button.on_click
        def _(_event):
            mode = next(key for key, label in MODES.items() if label == self._mode_input.value)
            self.request_selection(mode, self._weight_input.value)

    def request_selection(self, mode, mass):
        with self._lifecycle_lock:
            if self._closing or self._pending_mass is not None:
                return False
            try:
                selected = Selection(mode, mass)
            except (TypeError, ValueError, OverflowError) as exc:
                self._mass_notice.content = str(exc)
                return False
            if not super().request_mass(selected.mass_kg):
                return False
            self.requested_selection = selected
            self._mode_input.disabled = True
            return True


def run(args, world_factory=prepare_selection, viewer_class=ModeViewer):
    # Check both task contracts before starting the first physics world.
    empty_hand_cli.read_metadata(args.empty_checkpoint)
    box_metadata = cli._validate_checkpoint_support(hold_args(args, Selection("box", args.mass)))
    training_range = box_metadata["mass"]["mass_range_kg"]
    if not training_range or len(training_range) != 2:
        raise ValueError("Box checkpoint requires a recorded training mass range")
    world = world_factory(args, Selection(args.mode, args.mass))
    server = None
    history = []
    box_mass = args.mass
    notice = ""
    try:
        server = viser.ViserServer(host="127.0.0.1", port=args.port, label="K1 Load Lab · 空手 / 托箱")
        runtime = runtime_info(world.env.device)
        while True:
            revision = len(history)
            snapshot = world.snapshot(revision)
            history.append(snapshot)
            revision_dir = args.output / f"revision_{revision:03d}"
            write_json(revision_dir / "world.json", snapshot)
            write_json(args.output / "demo.json", {
                "runtime": runtime, "seed": args.seed, "port": args.port,
                "formal_evaluation": False, "mode_change_history": history,
                "note": "Apply rebuilds physics, switches to the matching policy, resets and pauses. Not continuous loading/unloading.",
            })
            viewer = viewer_class(world.wrapper, policy=lambda td: world.policy(dict(td)),
                mass=world.mass, selection=world.selection, box_mass=box_mass,
                training_range=training_range, notice=notice, support_mode="forearms",
                video_dir=revision_dir / "recordings", viser_server=server)
            viewer.pause()
            viewer.run(num_steps=round(args.seconds / world.env.step_dt))
            if viewer.requested_selection is None:
                break
            selected = viewer.requested_selection
            try:
                replacement = world_factory(args, selected)
            except Exception as exc:
                world.env.reset(seed=args.seed)
                notice = f"切换失败，已保留原场景并重置暂停。{type(exc).__name__}: {exc}"
            else:
                world.close()
                world = replacement
                if selected.mode == "box":
                    box_mass = selected.mass_kg
                notice = "已应用并暂停。点击 Play 播放。"
            # Windows redirected terminals may use cp1252. Keep console status
            # ASCII; the full Chinese message remains in the browser panel.
            print(ascii(notice), flush=True)
            # run() has joined the retiring scene/recording workers and removed
            # only its own client callbacks. The HTTP connection stays alive.
            server.gui.reset()
            server.scene.reset()
    finally:
        if server is not None:
            server.stop()
        world.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--mode", choices=tuple(MODES), default="empty")
    p.add_argument("--mass", type=cli._positive, default=DEFAULT_BOX_MASS)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--seconds", type=cli._positive, default=3600)
    p.add_argument("--port", type=cli._positive_int, default=8081)
    p.add_argument("--empty-checkpoint", type=Path, default=Path("shared_model/empty-hand-v1-20261030/policy.pt"))
    p.add_argument("--box-checkpoint", type=Path, default=Path("shared_model/stance-v2-20261029/policy.pt"))
    p.add_argument("--output", type=Path)
    return p


def main():
    args = build_parser().parse_args()
    Selection("box", args.mass)
    if args.port > 65535:
        raise ValueError("Port must be <= 65535")
    args.output = args.output or Path("outputs") / f"{datetime.now():%Y%m%d-%H%M%S-%f}_load_viewer"
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
