"""Install the isolated no-vision task without overwriting existing code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PACKAGE = Path(__file__).resolve().parents[1]
DEPLOY_COMMIT = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_package():
    manifest = json.loads((PACKAGE/"SHA256SUMS.json").read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Empty package manifest")
    for relative, expected in manifest.items():
        p = (PACKAGE/relative).resolve()
        if not p.is_relative_to(PACKAGE.resolve()) or not p.is_file():
            raise ValueError(f"Invalid or missing package path: {relative}")
        if sha(p) != expected["sha256"] or p.stat().st_size != expected["bytes"]:
            raise ValueError(f"Changed package file: {relative}")
    return manifest


def install(target):
    manifest = verify_package()
    target = Path(target).resolve(strict=True)
    top = Path(subprocess.check_output(["git","-C",str(target),"rev-parse","--show-toplevel"],text=True).strip()).resolve()
    if top != target or not (target/'scripts/deploy.py').is_file() or not (target/'booster_deploy/controllers/mujoco_controller.py').is_file():
        raise ValueError("--deploy-root must be the booster_deploy checkout top-level directory")
    revision = subprocess.check_output(["git","-C",str(target),"rev-parse","HEAD"],text=True).strip()
    if revision != DEPLOY_COMMIT:
        raise ValueError(f"Expected pinned booster_deploy {DEPLOY_COMMIT}, got {revision}. Use a separate checkout.")
    tracked_changes = subprocess.check_output(["git","-C",str(target),"diff","HEAD","--",
        "booster_deploy","scripts/deploy.py","tasks/__init__.py"],text=True)
    if tracked_changes.strip():
        raise ValueError("Upstream controller/launcher files differ from pinned revision; use a clean test checkout")
    source = PACKAGE/"tasks/team15_proprio"
    destination = target/"tasks/team15_proprio"
    files = sorted(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not files:
        raise ValueError("Package has no task files")
    for path in files:
        relative = path.relative_to(PACKAGE).as_posix()
        if relative not in manifest:
            raise ValueError(f"Task file absent from verified manifest: {relative}")
        out = destination/path.relative_to(source)
        if not out.resolve().is_relative_to(target) or out.is_symlink():
            raise ValueError(f"Unsafe destination: {out}")
        if out.exists() and (not out.is_file() or sha(out) != sha(path)):
            raise FileExistsError(f"Existing different file: {out}. Use a fresh test checkout.")
    for path in files:
        out = destination/path.relative_to(source)
        out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():
            shutil.copy2(path,out)
    return dict(task_directory=str(destination), deploy_commit=revision,
                task_files=len(files), hardware_control_enabled=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root",type=Path)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.verify_only:
        print(json.dumps(dict(verified_files=len(verify_package()))))
    elif args.deploy_root:
        print(json.dumps(install(args.deploy_root),indent=2))
    else:
        parser.error("Specify --deploy-root or --verify-only")


if __name__=="__main__":
    main()
