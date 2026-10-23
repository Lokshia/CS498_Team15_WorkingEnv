"""Device checks and provenance for comparable team runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "lifting_lab").glob("*.py"))
    files += [p for p in (root / "pyproject.toml", root / "uv.lock") if p.exists()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_device(requested):
    import torch
    if requested == "cpu":
        return "cpu"
    if requested not in ("cuda", "cuda:0"):
        raise ValueError("Choose cpu or cuda; GPU selection uses CUDA_VISIBLE_DEVICES.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the cu130 profile and check your NVIDIA driver. No silent CPU fallback.")
    return "cuda:0"


def runtime_info(device):
    import torch
    names = ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "booster-mjlab", "numpy", "viser")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    try:
        driver_query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=5, check=True)
        drivers = driver_query.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        drivers = []
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "nvidia_driver_inventory": drivers,
            "torch_cuda": torch.version.cuda, "versions": versions,
            "source_sha256": source_fingerprint()}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def doctor(device):
    import torch
    import warp as wp
    device = resolve_device(device)
    wp.init()
    warp_device = wp.get_device(device)
    x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    assert torch.isfinite(x @ x).all()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        assert warp_device.is_cuda
    info = runtime_info(device)
    info.update({"passed": True, "warp_device": str(warp_device),
                 "scope": "Runtime/device test only. Run smoke for actual batched physics."})
    print(json.dumps(info, indent=2))
    return info

