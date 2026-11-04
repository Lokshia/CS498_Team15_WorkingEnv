"""Install only the Team15 task into an existing, working booster_deploy checkout.

Does not install dependencies, edit upstream source, contact ROS, or start motors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REFERENCE_COMMIT = '7bb1462e2ac742eba5c01659b815e2e99fbe617f'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(deploy_root: Path, source: Path = HERE / 'tasks' / 'team15_hold'):
    root = deploy_root.resolve(strict=True)
    source = source.resolve(strict=True)
    required = ('scripts/deploy.py', 'booster_deploy/robots/k1.py',
                'booster_deploy/controllers/base_controller.py',
                'booster_deploy/controllers/booster_robot_controller.py',
                'booster_deploy/controllers/controller_cfg.py',
                'booster_deploy/utils/registry.py')
    for relative in required:
        if not (root / relative).is_file():
            raise ValueError(f'Not the expected booster_deploy root: missing {relative}')
    tasks = root / 'tasks'
    if not tasks.is_dir() or tasks.is_symlink():
        raise ValueError('Expected an existing, non-symlink tasks directory')
    destination = tasks / 'team15_hold'
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; nothing was overwritten. Compare or back it up explicitly.')
    files = sorted(p for p in source.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.onnx', '.npz'))
    if not files or not (source / '__init__.py').is_file() or not (source / 'models/proprio_hold.onnx').is_file():
        raise ValueError('Incomplete Team15 source package')
    before = {relative: sha(root / relative) for relative in required}
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # No recursive replacement/deletion. A failed copy leaves visible files for review.
    destination.mkdir()
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if sha(path) != sha(target):
            raise RuntimeError(f'Copied file hash mismatch: {target}')
    after = {relative: sha(root / relative) for relative in required}
    if before != after:
        raise RuntimeError('Upstream source changed during installation')
    receipt = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'k1_team15_hold', 'upstream_commit': revision,
        'reference_commit': REFERENCE_COMMIT,
        'matches_reference_commit': revision == REFERENCE_COMMIT,
        'upstream_source_sha256': before, 'upstream_source_unchanged': True,
        'installed_files': {p.relative_to(source).as_posix(): sha(p) for p in files},
        'robot_connected': False, 'hardware_validated': False,
    }
    with (destination / 'INSTALL_RECEIPT.json').open('x', encoding='utf-8') as file:
        json.dump(receipt, file, indent=2); file.write('\n')
    return receipt, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-root', type=Path, required=True)
    args = parser.parse_args()
    receipt, destination = install(args.deploy_root)
    print(f'Installed: {destination}')
    print('Upstream controller / deploy.py: unchanged')
    if not receipt['matches_reference_commit']:
        print('Note: upstream revision differs or is unknown; run the offline check before using this task.')
    print('Next, from booster_deploy root:')
    print('  python3 scripts/deploy.py --list')
    print('  python3 -m tasks.team15_hold.check')
    print('No robot control was started.')


if __name__ == '__main__':
    main()
