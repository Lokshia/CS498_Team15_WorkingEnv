"""Independent empty-hand standing task. No payload body or synthetic box input.

Actor: 72 IMU/proprioception/previous-action features. Critic adds simulated
base linear velocity and height (76). Same 22 joint-position actions / 50 Hz
as the forearm task. This is a simulation task, not a hardware adapter.
"""
from copy import deepcopy
import math
import torch
from mjlab.envs import mdp as builtin
from mjlab.envs.mdp.events import reset_scene_to_default, resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul
from .env_cfg import make_env_cfg
from . import mdp
from .model import robot_mass_kg

TASK_SCHEMA = 'k1-empty-hand-stand-v1'
ACTOR_SCHEMA = 'k1-imu-proprio-last-action-72-v1'
CRITIC_SCHEMA = 'k1-empty-hand-privileged-76-v1'


def observation(env, privileged=False):
    r = env.scene['robot'].data
    parts = [r.root_link_ang_vel_b, r.projected_gravity_b,
             r.joint_pos-r.default_joint_pos, r.joint_vel, env.action_manager.action]
    if privileged:
        parts += [r.root_link_lin_vel_b,
                  (r.root_link_pos_w[:, 2]-env.scene.env_origins[:, 2])[:, None]]
    return torch.cat(parts, dim=-1)


def reset(env, env_ids, joint_jitter=.015, speed_range=.02):
    ids = resolve_env_ids(env, env_ids)
    reset_scene_to_default(env, ids)
    n = len(ids); robot = env.scene['robot']; d = robot.data
    uniform = lambda shape, scale: (2*torch.rand(shape, device=env.device)-1)*scale
    yaw = uniform((n,), .05); zero = torch.zeros_like(yaw)
    rot = quat_from_euler_xyz(zero, zero, yaw)
    state = d.default_root_state[ids].clone()
    state[:, :3] = quat_apply(rot, state[:, :3]) + env.scene.env_origins[ids]
    state[:, :2] += uniform((n, 2), .05)
    state[:, 3:7] = quat_mul(rot, state[:, 3:7])
    state[:, 7:9] = uniform((n, 2), speed_range)
    robot.write_root_state_to_sim(state, ids)
    robot.write_joint_state_to_sim(d.default_joint_pos[ids]+uniform((n, d.joint_pos.shape[1]), joint_jitter),
                                  torch.zeros_like(d.joint_vel[ids]), env_ids=ids)


def feet(env):
    if not hasattr(env, '_empty_sole_ids'):
        env._empty_sole_ids = env.scene['robot'].find_sites(
            ('left_foot_sole', 'right_foot_sole'), preserve_order=True)[0]
        names = [n.split('/')[-1] for n in env.scene['feet_ground_contact'].primary_names]
        if len(names) != 2 or set(names) != {'left_foot_link','right_foot_link'}:
            raise ValueError('Expected exactly two named foot contact primaries')
        env._empty_contact_ids = [names.index(n) for n in ('left_foot_link','right_foot_link')]
    r = env.scene['robot'].data; sensor = env.scene['feet_ground_contact'].data
    q = r.site_quat_w[:, env._empty_sole_ids]
    z = torch.zeros_like(q[..., :3]); z[..., 2] = 1
    normal_z = quat_apply(q,z)[..., 2].clamp(-1,1)
    force = -sensor.force.reshape(env.num_envs,2,3)[:,env._empty_contact_ids]
    contact = sensor.found.reshape(env.num_envs,2)[:,env._empty_contact_ids] > 0
    return normal_z, force, contact


def feet_flat(env):
    return (1-feet(env)[0]).sum(-1)


def missing_feet(env):
    return (~feet(env)[2]).float().sum(-1)


def load_share(env):
    _, force, contact = feet(env)
    vertical = force[...,2].clamp_min(0)*contact
    total = vertical.sum(-1)
    share = vertical/total[:,None].clamp_min(1e-6)
    cost = (torch.relu(.3-share)/.3).square().sum(-1)
    return torch.where(total > .1*robot_mass_kg()*9.81, cost, torch.ones_like(cost))


def foot_slip(env):
    _, _, contact = feet(env)
    v = env.scene['robot'].data.site_lin_vel_w[:,env._empty_sole_ids,:2]
    return (v.square().sum(-1)*contact).sum(-1)


def posture(env):
    r = env.scene['robot'].data
    return ((r.joint_pos-r.default_joint_pos)/.25).square().mean(-1)


def height_reward(env):
    r = env.scene['robot'].data
    error = r.root_link_pos_w[:,2]-env.scene.env_origins[:,2]-r.default_root_state[:,2]
    return torch.exp(-(error/.05).square())


def base_motion(env):
    r = env.scene['robot'].data
    return r.root_link_lin_vel_b.square().sum(-1)+.2*r.root_link_ang_vel_b.square().sum(-1)


def fell(env):
    r = env.scene['robot'].data
    return ((r.root_link_pos_w[:,2]-env.scene.env_origins[:,2] < .38)
            | (mdp.robot_upright(env) < math.cos(math.radians(45))))


def make_empty_hand_cfg(num_envs=256, seconds=20, seed=11, auto_reset=True,
                        joint_jitter=.015, speed_range=.02, push_interval=None, push_speed=0.):
    for value in (joint_jitter, speed_range, push_speed):
        if not math.isfinite(value) or value < 0: raise ValueError('Invalid reset/push magnitude')
    if (push_interval is None) != (push_speed == 0): raise ValueError('Push interval and speed must be provided together')
    cfg = make_env_cfg(payload_mass_kg=0, num_envs=num_envs, episode_length_s=seconds,
                       support_mode='forearms', reward_profile='stance-v2')
    # Only config construction reuses the forearm defaults. The compiled scene
    # below has no payload entity, contact sensors, equality or ghost rigid body.
    cfg.scene.entities.pop('payload')
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name == 'feet_ground_contact')
    cfg.scene.spec_fn = None
    cfg.support_mode='empty_hand'; cfg.reward_profile='empty-hand-v1'
    cfg.requested_payload_mass_kg=cfg.actual_payload_mass_kg=0.
    cfg.seed=seed; cfg.auto_reset=auto_reset
    cfg.events={'reset':EventTermCfg(func=reset, mode='reset', params={'joint_jitter':joint_jitter,'speed_range':speed_range})}
    if push_interval is not None:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        if len(push_interval)!=2 or not 0 < push_interval[0] <= push_interval[1]:
            raise ValueError('Invalid push interval')
        cfg.events['push']=EventTermCfg(func=builtin.push_by_setting_velocity,mode='interval',interval_range_s=tuple(push_interval),
            params={'velocity_range':{'x':(-push_speed,push_speed),'y':(-push_speed,push_speed)},'asset_cfg':SceneEntityCfg('robot')})
    cfg.observations={name:ObservationGroupCfg(terms={'state':ObservationTermCfg(func=observation,params={'privileged':privileged})},
        concatenate_terms=True, enable_corruption=False) for name,privileged in [('actor',False),('critic',True)]}
    cfg.rewards={
        'alive':RewardTermCfg(func=builtin.is_alive,weight=1.),
        'height':RewardTermCfg(func=height_reward,weight=3.),
        'upright':RewardTermCfg(func=mdp.trunk_tilt_cost,weight=-1.5),
        'feet_flat':RewardTermCfg(func=feet_flat,weight=-20.),
        'missing_feet':RewardTermCfg(func=missing_feet,weight=-2.),
        'load_share':RewardTermCfg(func=load_share,weight=-.5),
        'posture':RewardTermCfg(func=posture,weight=-1.),
        'leg_symmetry':RewardTermCfg(func=mdp.leg_symmetry_cost,weight=-.5),
        'base_motion':RewardTermCfg(func=base_motion,weight=-4.),
        'action_rate':RewardTermCfg(func=builtin.action_rate_l2,weight=-.15),
        'joint_velocity':RewardTermCfg(func=mdp.joint_velocity_cost,weight=-.002),
        'foot_slip':RewardTermCfg(func=foot_slip,weight=-.5),
        'effort':RewardTermCfg(func=mdp.effort_cost,weight=-.00002),
        'joint_limits':RewardTermCfg(func=builtin.joint_pos_limits,weight=-.1),
        'failure':RewardTermCfg(func=builtin.is_terminated,weight=-100.),
    }
    cfg.terminations={'time_out':TerminationTermCfg(func=builtin.time_out,time_out=True),
                      'fell':TerminationTermCfg(func=fell),'nonfinite':TerminationTermCfg(func=builtin.nan_detection)}
    return cfg
