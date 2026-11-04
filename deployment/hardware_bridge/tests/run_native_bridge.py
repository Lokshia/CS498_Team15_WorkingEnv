"""Test the NEW calibrated/rate-limited action path in the frozen native scene.

This does not exercise real ROS, motors, preparation, or loading. Profile is
derived from simulation physics and is explicitly refused by --live.
"""
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
PACKAGE = BRIDGE.parent / 'team15_proprio_package'
sys.path.insert(0, str(BRIDGE))
from fixtures import make_config
from team15_hardware.core import Actor, ControlCore
from team15_hardware.config import validate_config
from team15_hardware.ros_transport import StateSnapshot

MODEL = PACKAGE/'tasks/team15_proprio/models/proprio_hold.onnx'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-root', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--mass-kg', type=float, default=.5)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds < 20 or not np.isfinite([args.seconds,args.mass_kg]).all() or args.mass_kg < .001:
        parser.error('Need at least one >=20-second episode and a finite physical mass')
    args.output.mkdir(parents=True, exist_ok=False)
    spec=importlib.util.spec_from_file_location('frozen_validator',PACKAGE/'scripts/validate_package.py')
    validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    validator.verify_checkout(args.deploy_root,validator.DEPLOY_COMMIT)
    validator.verify_checkout(args.assets_root,validator.ASSETS_COMMIT)
    sys.path.insert(0,str(args.assets_root/'src')); sys.path.insert(0,str(args.deploy_root))
    import torch
    import mujoco
    torch.set_num_threads(1)
    import tasks.team15_proprio
    from booster_deploy.utils.registry import get_task
    actor=Actor(MODEL)
    exports=actor.verify_vectors(MODEL.with_name('proprio_hold.test_vectors.npz'))
    profiles=[]

    class BridgePolicy:
        def __init__(self, cfg, controller):
            self.controller=controller
            self.last_raw_action=torch.zeros(22)
            self.last_action=torch.zeros(22)

        def snapshot(self):
            c=self.controller; d=c.robot.data
            quat=d.root_quat_w.numpy().astype(float)
            rotation=np.empty(9); mujoco.mju_quat2Mat(rotation,quat); r=rotation.reshape(3,3)
            rpy=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1)),np.arctan2(r[1,0],r[0,0])]
            return StateSnapshot(q=tuple(d.joint_pos.tolist()),dq=tuple(d.joint_vel.tolist()),
                                 tau=tuple(d.feedback_torque.tolist()),rpy=tuple(rpy),gyro=tuple(d.root_ang_vel_b.tolist()),
                                 received_monotonic=float(c.mj_data.time),sequence=round(c.mj_data.time/.02))

        def reset(self):
            c=self.controller
            profile=make_config(actor.contract)
            for i,row in enumerate(profile['joints']):
                joint=c.mj_model.joint(row['name'])
                row.update(model_min_rad=float(c.mj_model.jnt_range[joint.id,0]),
                           model_max_rad=float(c.mj_model.jnt_range[joint.id,1]),
                           kp=float(c.cfg.robot.joint_stiffness[i]),kd=float(c.cfg.robot.joint_damping[i]),
                           measured_torque_limit_nm=1.1*float(c.cfg.robot.effort_limit[i]),
                           measured_speed_limit_rad_s=10.,tracking_error_limit_rad=1.,target_rate_limit_rad_s=2.)
            profile['limits']['run_timeout_s']=args.seconds
            profile=validate_config(profile,actor.contract)
            profiles.append(profile)
            self.core=ControlCore(profile,actor.contract)
            prime=self.core.prime(self.snapshot(),float(c.mj_data.time))
            self.core.commit_sent(prime,float(c.mj_data.time))

        def compute_observation(self):
            return torch.tensor(self.core.observation(self.snapshot(),float(self.controller.mj_data.time)))

        def inference(self):
            now=float(self.controller.mj_data.time)
            action=actor(self.compute_observation().numpy())
            command=self.core.policy_command(action,self.snapshot(),now)
            self.core.commit_sent(command,now)
            self.last_raw_action=torch.tensor(action,dtype=torch.float32)
            self.last_action=torch.tensor(command.effective_action,dtype=torch.float32)
            return torch.tensor(command.serial_q,dtype=torch.float32)

        def projection_statistics(self):
            return {'path':'new hardware ControlCore under simulation-only profile',
                    'per_joint_target_rate_limit_rad_s':2.,'previous_action':'effective sent target',
                    'real_robot_validated':False}

    def factory():
        cfg=deepcopy(get_task(validator.TASK_NAME))
        cfg.policy.constructor=BridgePolicy
        return cfg

    def hashes():
        paths=list((BRIDGE/'team15_hardware').glob('*.py'))+[Path(__file__),HERE/'fixtures.py',MODEL,PACKAGE/'scripts/validate_package.py']
        return {str(p.relative_to(BRIDGE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before=hashes()
    report={'schema':'team15-hardware-bridge-native-validation-v1','hardware_validated':False,
            'ros_commands_sent':0,'exports':exports,'episodes':[],'source_hashes':before,
            'limitations':['No ROS or real motors. Simulation-only gains and calibrated identity mapping.',
                           'Arms already raised and box supported. No physical preparation or loading validation.',
                           '2 rad/s rate limit is a simulation test value, not a motor recommendation.',
                           'Unchanged frozen strict scoring and original full episodes.']}
    for i in range(args.episodes):
        summary,rows,audit=validator.rollout(factory,args.seconds,31031+i,args.mass_kg,i>0)
        report['episodes'].append(summary);report['model_audit']=audit
        if rows:
            np.savez_compressed(args.output/f'episode_{i:02d}.npz',**{k:np.asarray([row[k] for row in rows]) for k in rows[0]})
        print(json.dumps({k:summary.get(k) for k in ['seed','steps_completed','failure','strict_standing_pass','all_step_max_payload_tilt_deg']}),flush=True)
    report['source_integrity_passed']=before==hashes()
    report['all_passed']=report['source_integrity_passed'] and all(e['strict_standing_pass'] and e['commanded_targets_within_xml_limits'] for e in report['episodes'])
    (args.output/'simulation_only_profile.json').write_text(json.dumps(profiles[0],indent=2),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"all_passed={report['all_passed']}")
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
