"""Train, evaluate and view the separate empty-hand task; simulation only."""
from pathlib import Path
import argparse
import json
import math
from copy import deepcopy
import numpy as np
import torch
from . import empty_hand as task
from .runtime import runtime_info, resolve_device, write_json, file_sha256


def metadata_contract():
    return dict(task_schema=task.TASK_SCHEMA, actor_observation_schema=task.ACTOR_SCHEMA,
        critic_observation_schema=task.CRITIC_SCHEMA, actor_observation_dim=72,
        critic_observation_dim=76, action_dim=22, control_dt_s=.02, payload_entity_present=False,
        reward_profile='empty-hand-v1', action_contract='original K1 joint-position scale, default-offset, clipped [-1,1]')


def read_metadata(checkpoint):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    data=json.loads(checkpoint.with_name('experiment.json').read_text(encoding='utf-8'))
    for key,value in metadata_contract().items():
        if data.get(key)!=value: raise ValueError(f'Incompatible empty-hand checkpoint {key}: {data.get(key)!r} != {value!r}')
    if 'runner' not in data: raise ValueError('Missing runner configuration')
    return data


def environment(a, *, training=False):
    from mjlab.envs import ManagerBasedRlEnv
    cfg=task.make_empty_hand_cfg(num_envs=a.num_envs,seconds=20 if training else a.seconds+1,
        seed=a.seed,auto_reset=training,joint_jitter=a.joint_jitter,speed_range=a.reset_speed,
        push_interval=(5.,5.) if a.push_speed else None,push_speed=a.push_speed)
    if a.command=='demo':
        # Keep failure termination in evaluation. Interactive viewer can continue
        # showing a failed posture, so it is never counted as a successful test.
        cfg.terminations={k:v for k,v in cfg.terminations.items() if k in ('time_out','nonfinite')}
    env=ManagerBasedRlEnv(cfg,device=resolve_device(a.device))
    model=env.sim.mj_model
    assert set(env.scene.entities)=={'robot','terrain'} and model.neq==0
    assert not any('payload' in (model.body(i).name or '') for i in range(model.nbody))
    assert math.isclose(float(model.body_mass.sum()),task.robot_mass_kg(),rel_tol=1e-6)
    return env


def make_runner(env,cfg,log_dir=None):
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env,clip_actions=1.),cfg,log_dir=log_dir,device=env.device)
    runner.logger.git_status_repos=[]
    return runner


def policy(env,checkpoint):
    if checkpoint is None:
        return lambda obs: torch.zeros((env.num_envs,22),device=env.device)
    from tensordict import TensorDict
    metadata=read_metadata(checkpoint)
    runner=make_runner(env,metadata['runner'])
    runner.load(str(checkpoint),map_location=env.device)
    actor=runner.get_inference_policy(device=env.device)
    return lambda obs: actor(TensorDict(obs,batch_size=[env.num_envs])).clamp(-1,1)


def train(a):
    from .cli import _runner_config, _assert_finite_training_state, _apply_learning_rate_override
    old=read_metadata(a.resume) if a.resume else None
    if old and old['seed']!=a.seed: raise ValueError('Resume must keep training seed')
    env=environment(a,training=True)
    try:
        cfg=deepcopy(old['runner']) if old else _runner_config(a.seed)
        cfg.update(max_iterations=a.iterations, num_steps_per_env=24,save_interval=50)
        cfg['algorithm'].update(learning_rate=a.learning_rate,entropy_coef=.001)
        if not old: cfg['actor']['distribution_cfg'].update(init_std=.15)
        runner=make_runner(env,cfg,str(a.output))
        start=0; prior=0
        if old:
            runner.load(str(a.resume),map_location=env.device)
            runner.current_learning_iteration+=1
            start=runner.current_learning_iteration
            prior=old['prior_env_steps']+(start-old['phase_start_iteration'])*old['num_envs']*24
        else:
            # Start the mean at zero-action PD, with stochastic exploration.
            layers=[x for x in runner.alg.actor.mlp.modules() if isinstance(x,torch.nn.Linear)]
            torch.nn.init.zeros_(layers[-1].weight);torch.nn.init.zeros_(layers[-1].bias)
        _apply_learning_rate_override(runner,a)
        metadata={**metadata_contract(),'runtime':runtime_info(env.device),'seed':a.seed,'runner':cfg,
            'num_envs':a.num_envs,'phase_start_iteration':start,'prior_env_steps':prior,
            'planned_phase_steps':a.iterations*24*a.num_envs,'reset_joint_jitter_rad':a.joint_jitter,
            'reset_speed_m_s':a.reset_speed,'push_speed_m_s':a.push_speed,
            'initialization':'fresh network with zero mean output, std=.15; no hold weights' if not old else 'resume same empty-hand schema, optimizer and normalizers; environment/RNG restart',
            'resume_from':str(a.resume) if a.resume else None,'resume_sha256':file_sha256(a.resume) if a.resume else None,
            'hardware_ready':False}
        write_json(a.output/'experiment.json',metadata)
        update=runner.alg.update
        def checked_update():
            losses=update(); _assert_finite_training_state(losses); return losses
        runner.alg.update=checked_update
        runner.learn(a.iterations,init_at_random_ep_len=False)
        _assert_finite_training_state(runner.alg.save())
        count=runner.current_learning_iteration-start+1
        assert count==a.iterations
        runner.save(str(a.output/'policy.pt'))
        metadata.update(last_completed_iteration=runner.current_learning_iteration,
                        actual_phase_steps=count*24*a.num_envs,actual_cumulative_steps=prior+count*24*a.num_envs)
        metadata['runtime_source_sha256_at_end']=runtime_info(env.device)['source_sha256']
        assert metadata['runtime']['source_sha256']==metadata['runtime_source_sha256_at_end']
        write_json(a.output/'experiment.json',metadata)
        write_json(a.output/'completed.json',dict(checkpoint_sha256=file_sha256(a.output/'policy.pt'),
            finite_training_state=True,iterations=count,steps=metadata['actual_cumulative_steps'],quality='requires separate evaluation'))
    finally:env.close()


def evaluate(a):
    if a.checkpoint: read_metadata(a.checkpoint)
    env=environment(a)
    try:
        actor=policy(env,a.checkpoint);obs,_=env.reset(seed=a.seed)
        assert obs['actor'].shape==(a.num_envs,72) and obs['critic'].shape==(a.num_envs,76)
        runtime=runtime_info(env.device); r=env.scene['robot']; snap=lambda x:x.detach().cpu().numpy().copy()
        origin=snap(r.data.root_link_pos_w[:,:2]); active=np.ones(a.num_envs,bool)
        failed=np.zeros(a.num_envs,bool); rows=[];masks=[]
        arm_ids=r.find_joints(('Left_Shoulder_.*','Left_Elbow_.*','Right_Shoulder_.*','Right_Elbow_.*'),preserve_order=True)[0]
        assert len(arm_ids)==8
        required=round(a.seconds/env.step_dt)
        assert math.isclose(required*env.step_dt,a.seconds)
        with torch.inference_mode():
            for i in range(required):
                mask=active.copy(); action=actor(obs)
                if not torch.isfinite(action).all(): raise ValueError('Nonfinite policy action')
                obs,reward,term,trunc,_=env.step(action)
                nz,force,contact=task.feet(env)
                row=dict(time_s=(i+1)*env.step_dt,force_n=snap(force),contact=snap(contact),
                    foot_tilt_deg=np.degrees(np.arccos(np.clip(snap(nz),-1,1))),
                    trunk_tilt_deg=np.degrees(np.arccos(np.clip(snap(task.mdp.robot_upright(env)),-1,1))),
                    q=snap(r.data.joint_pos),qd=snap(r.data.joint_vel),torque=snap(r.data.qfrc_actuator),
                    arm_error_rad=snap((r.data.joint_pos-r.data.default_joint_pos)[:,arm_ids]),
                    xy_drift_m=np.linalg.norm(snap(r.data.root_link_pos_w[:,:2])-origin,axis=-1),
                    height_m=snap(r.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]))
                if not all(np.isfinite(v[mask]).all() for k,v in row.items() if k!='time_s'):raise ValueError('Nonfinite active diagnostic')
                rows.append(row);masks.append(mask)
                done=snap(term|trunc).astype(bool);failed[mask&done]=True;active &= ~done
                if not active.any():break
                reset_ids=(term|trunc).nonzero(as_tuple=False).flatten()
                if reset_ids.numel():obs,_=env.reset(env_ids=reset_ids)
        arrays={k:np.asarray([x[k] for x in rows]) for k in rows[0]};masks=np.asarray(masks);episodes=[]
        for i in range(a.num_envs):
            valid=masks[:,i];t=arrays['time_s'][valid];steady=t>=2
            complete=len(t)==required and not failed[i]
            metrics=None
            if steady.any():
                selected=lambda k: arrays[k][valid,i][steady]
                force=selected('force_n')[:,:,2];positive=np.maximum(force,0)
                metrics=dict(foot_tilt_mean_deg=selected('foot_tilt_deg').mean(0).tolist(),
                    foot_tilt_max_deg=selected('foot_tilt_deg').max(0).tolist(),
                    flat_contact_fraction=float((selected('contact').all(-1)&(selected('foot_tilt_deg')<5).all(-1)).mean()),
                    ground_reaction_mean_n=force.mean(0).tolist(),
                    left_load_share=float((positive[:,0]/positive.sum(-1).clip(1e-6)).mean()),
                    trunk_tilt_mean_deg=float(selected('trunk_tilt_deg').mean()),
                    trunk_tilt_p95_deg=float(np.quantile(selected('trunk_tilt_deg'),.95)),
                    joint_velocity_rms_rad_s=float(np.sqrt(np.mean(selected('qd')**2))),
                    joint_torque_rms_nm=float(np.sqrt(np.mean(selected('torque')**2))),
                    arm_pose_rms_rad=float(np.sqrt(np.mean(selected('arm_error_rad')**2))))
            drift=float(arrays['xy_drift_m'][valid,i].max())
            success=bool(complete and metrics and metrics['flat_contact_fraction']>=.95
                and metrics['trunk_tilt_p95_deg']<5 and drift<.10
                and metrics['arm_pose_rms_rad']<.15 and metrics['joint_velocity_rms_rad_s']<.2)
            episodes.append(dict(slot=i,recorded_seconds=float(t[-1]),completed_without_fall=bool(complete),
                success=success,max_xy_drift_m=drift,steady=metrics))
        end=runtime_info(env.device);assert runtime['source_sha256']==end['source_sha256']
        summary=dict(**metadata_contract(),runtime=runtime,runtime_source_sha256_at_end=end['source_sha256'],
            checkpoint=str(a.checkpoint) if a.checkpoint else None,
            checkpoint_sha256=file_sha256(a.checkpoint) if a.checkpoint else None,
            policy='trained deterministic policy' if a.checkpoint else 'zero-action position PD baseline',
            seed=a.seed,num_envs=a.num_envs,seconds=a.seconds,joint_names=r.joint_names,
            reset_joint_jitter_rad=a.joint_jitter,reset_speed_m_s=a.reset_speed,push_speed_m_s=a.push_speed,
            successes=sum(e['success'] for e in episodes),completed_without_fall=sum(e['completed_without_fall'] for e in episodes),
            criteria='Full duration; at t>=2s >=95% both feet contact+sole tilts<5deg, trunk p95<5deg, arm pose RMS<.15rad, joint speed RMS<.2rad/s; whole-episode base XY drift<.10m. No reset episode replacement.',
            sampling='50Hz last physics substep, not force substep average',episodes=episodes,
            limitations=['Simulation only; no hardware interface or safety acceptance.', 'Arms-forward initial posture, no arms-down transition or adding/removing a box.'])
        np.savez_compressed(a.output/'trace.npz',**arrays,active_original_episode=masks)
        write_json(a.output/'evaluation.json',summary)
        print(json.dumps({k:summary[k] for k in ('successes','completed_without_fall','num_envs')},indent=2),flush=True)
    finally:env.close()


def demo(a):
    import viser
    from booster_mjlab.viewer import RecordingViserPlayViewer
    from mjlab.rl import RslRlVecEnvWrapper
    env=environment(a);server=None
    try:
        actor=policy(env,a.checkpoint);wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        env.reset(seed=a.seed)
        server=viser.ViserServer(host='127.0.0.1',port=a.port,label='K1 Empty-hand Lab')
        class EmptyViewer(RecordingViserPlayViewer):
            def setup(self):
                self._server.gui.add_markdown('**真正空手 / No payload**\n\n独立空手策略，72维本体观测。没有箱子实体、质量代理或虚拟箱子输入。\n\n仿真演示；不支持放箱/卸箱或直接真机部署。')
                super().setup()
                with self._sim_lock,self._server.atomic():
                    self._scene.paused=self._is_paused
                    self._scene.update(self.env.unwrapped.sim.data);self._server.flush()
        viewer=EmptyViewer(wrapper,policy=lambda td:actor(dict(td)),video_dir=a.output/'recordings',viser_server=server)
        if a.paused: viewer.pause()
        write_json(a.output/'demo.json',dict(**metadata_contract(),runtime=runtime_info(env.device),
            checkpoint=str(a.checkpoint),checkpoint_sha256=file_sha256(a.checkpoint) if a.checkpoint else None,
            seed=a.seed,port=a.port,formal_evaluation=False))
        viewer.run(num_steps=round(a.seconds/env.step_dt))
    finally:
        env.close()
        if server:server.stop()


def main():
    from .cli import _positive,_nonnegative,_positive_int
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('train','evaluate','demo'):
        s=sub.add_parser(name);s.add_argument('--device',choices=('cpu','cuda'),default='cpu' if name=='demo' else 'cuda')
        s.add_argument('--num-envs',type=_positive_int,default=1 if name=='demo' else 256 if name=='train' else 16)
        s.add_argument('--seed',type=int,default=11 if name=='train' else 5011)
        s.add_argument('--seconds',type=_positive,default=20)
        s.add_argument('--joint-jitter',type=_nonnegative,default=.015)
        s.add_argument('--reset-speed',type=_nonnegative,default=.02)
        s.add_argument('--push-speed',type=_nonnegative,default=0)
        s.add_argument('--output',type=Path,required=True)
        if name=='train':
            s.add_argument('--iterations',type=_positive_int,default=300)
            s.add_argument('--learning-rate',type=_positive,default=.0003)
            s.add_argument('--resume',type=Path)
        else:s.add_argument('--checkpoint',type=Path)
        if name=='demo':
            s.add_argument('--port',type=_positive_int,default=8081);s.add_argument('--paused',action='store_true')
    a=p.parse_args()
    if a.command=='demo' and a.num_envs!=1:p.error('Viewer requires one environment')
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    {'train':train,'evaluate':evaluate,'demo':demo}[a.command](a)


if __name__=='__main__':main()
