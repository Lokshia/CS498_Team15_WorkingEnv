"""One no-box native MuJoCo diagnostic; does not validate hardware or loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-root', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['TEAM15_LOG_DIR'] = str(args.output/'policy_logs')
    sys.path.insert(0, str(args.assets_root/'src')); sys.path.insert(0, str(args.deploy_root))
    import torch
    torch.set_num_threads(1)
    import tasks.team15_hold
    from booster_deploy.utils.registry import get_task
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from tasks.team15_hold.contract import TASK_NAME
    controller = MujocoController(get_task(TASK_NAME).copy())
    controller.update_state(); controller.start()
    origin = controller.mj_data.qpos[:2].copy()
    rows=[]; failure=None
    try:
        for step in range(1000):
            controller.update_state()
            target = controller.policy_step()
            controller.ctrl_step(target)
            controller.update_state()
            quat = controller.robot.data.root_quat_w.numpy()
            from booster_deploy.utils.isaaclab.math import quat_apply_inverse
            gravity = quat_apply_inverse(torch.tensor(quat), torch.tensor([0.,0.,-1.])).numpy()
            height = float(controller.mj_data.qpos[2])
            tilt = float(np.rad2deg(np.arccos(np.clip(-gravity[2],-1.,1.))))
            row={'time_s':float(controller.mj_data.time),'base_height_m':height,'trunk_tilt_deg':tilt,
                 'drift_m':float(np.linalg.norm(controller.mj_data.qpos[:2]-origin))}
            rows.append(row)
            if not np.isfinite(controller.mj_data.qpos).all() or height<.28 or tilt>45:
                failure='fall_or_invalid_state'; break
    except (RuntimeError, ValueError) as exc:
        failure=f'{type(exc).__name__}: {exc}'
    finally:
        controller.stop(); controller.policy.close()
    task_root = args.deploy_root/'tasks/team15_hold'
    report={'schema':'team15-native-task-empty-simulation-v1','task':TASK_NAME,
            'seconds_requested':20,'steps_expected':1000,'steps_completed':len(rows),'failure':failure,
            'no_box_present':True,'nq':int(controller.mj_model.nq),'nv':int(controller.mj_model.nv),
            'basic_upright_diagnostic_passed':failure is None and len(rows)==1000,
            'max_trunk_tilt_deg':max((r['trunk_tilt_deg'] for r in rows), default=None),
            'min_base_height_m':min((r['base_height_m'] for r in rows), default=None),
            'max_drift_m':max((r['drift_m'] for r in rows), default=None),
            'source_sha256':{p.relative_to(task_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in task_root.rglob('*.py') if '__pycache__' not in p.parts},
            'limitations':['Single original episode; starts in raised-arm nominal pose, no robot preparation.',
                           'Basic height/tilt diagnostic only; not the full loaded-box standing score.',
                           'No hardware/ROS commands; no general empty-hand robustness claim.'],
            'hardware_validated':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (args.output/'trace.json').write_text(json.dumps(rows)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    return 0 if report['basic_upright_diagnostic_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
