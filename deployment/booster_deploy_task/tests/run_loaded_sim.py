"""Run the native task Policy in the existing free-box simulator, not on hardware."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-root', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--validated-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['TEAM15_LOG_DIR'] = str(args.output/'policy_logs')
    package = args.validated_package.resolve(strict=True)
    spec = importlib.util.spec_from_file_location('frozen_validator', package/'scripts/validate_package.py')
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    validator.verify_checkout(args.deploy_root, validator.DEPLOY_COMMIT)
    validator.verify_checkout(args.assets_root, validator.ASSETS_COMMIT)
    sys.path.insert(0, str(args.assets_root/'src')); sys.path.insert(0, str(args.deploy_root))
    import torch
    torch.set_num_threads(1)
    import tasks
    tasks.__path__.append(str(package/'tasks'))
    import tasks.team15_hold
    import tasks.team15_proprio
    from booster_deploy.utils.registry import get_task
    from tasks.team15_hold.contract import TASK_NAME

    def factory():
        return deepcopy(get_task(TASK_NAME))

    task_root = args.deploy_root/'tasks/team15_hold'
    paths = sorted(p for p in task_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    def hashes():
        return {p.relative_to(task_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    report = {'schema':'team15-native-task-loaded-simulation-v1', 'task':TASK_NAME,
              'real_robot_validated':False, 'ros_commands_sent':0, 'source_sha256':before,
              'limitations':['Same new task Policy in explicit-address free-box simulator.',
                             'Original deploy.py and ROS control were not run on a robot.',
                             'Starts at raised-arm pose with box supported; preparation/loading not tested.',
                             'Uses unchanged frozen strict scoring; every original episode reported.'],
              'episodes':[]}
    for i in range(3):
        summary, rows, audit = validator.rollout(factory, 20., 32031+i, .5, i>0)
        report['episodes'].append(summary); report['model_audit'] = audit
        if rows:
            np.savez_compressed(args.output/f'episode_{i:02d}.npz',
                                **{key:np.asarray([row[key] for row in rows]) for key in rows[0]})
        print(json.dumps({k:summary.get(k) for k in ('seed','steps_completed','failure','strict_standing_pass','all_step_max_payload_tilt_deg')}), flush=True)
    report['source_unchanged'] = before == hashes()
    report['all_passed'] = report['source_unchanged'] and all(
        e['strict_standing_pass'] and e['commanded_targets_within_xml_limits'] for e in report['episodes'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('all_passed='+str(report['all_passed']))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
