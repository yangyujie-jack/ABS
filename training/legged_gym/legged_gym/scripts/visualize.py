import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import onnx
from onnx2pytorch import ConvertModel


def observation_distribution():
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough', 'exported')
    obs = np.load(os.path.join(path, 'ra_obs.npy'))
    obs = obs.reshape((-1, obs.shape[-1]))
    for i in range(obs.shape[1]):
        plt.figure()
        plt.hist(obs[:, i], bins=50)
        plt.savefig(os.path.join(path, f'ra_obs_{i}.png'))
        plt.close()


def collect_trajectory(args, traj_len: int = 200):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = args.num_envs
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_types = ['flat']  # do not duplicate!
    env_cfg.terrain.terrain_proportions = [0.5]
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.0 # allow illusion
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [-0.2, -0.2]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.domain_rand.erfi = False

    env_cfg.domain_rand.init_x_range = [-1., 0.]
    env_cfg.domain_rand.init_y_range = [0., 1.]
    env_cfg.domain_rand.init_yaw_range = [-np.pi / 4, -np.pi / 4]
    env_cfg.domain_rand.init_vlinx_range = [np.sqrt(2), np.sqrt(2)]
    env_cfg.domain_rand.init_vliny_range = [-np.sqrt(2), -np.sqrt(2)]
    env_cfg.domain_rand.init_vlinz_range = [0., 0.]
    env_cfg.domain_rand.init_vang_range = [0., 0.]

    env_cfg.init_state.pos = [0.0, 0.0, 0.32]
    env_cfg.domain_rand.init_dof_factor = [1.0, 1.0]
    env_cfg.domain_rand.stand_bias3 = [0.0, 0.2, -0.3]
    env_cfg.asset.object_num = 1
    env_cfg.asset.test_mode = True
    env_cfg.asset.test_obj_pos = [[[1.],[1.]]]
    env_cfg.asset.test_obj_pos = torch.Tensor(env_cfg.asset.test_obj_pos).to('cuda')
    env_cfg.asset.test_obj_pos = env_cfg.asset.test_obj_pos[:,:,:env_cfg.asset.object_num]
    env_cfg.asset.test_obj_pos = env_cfg.asset.test_obj_pos.repeat(env_cfg.env.num_envs,1,1)
    env_cfg.commands.ranges.use_polar = False
    env_cfg.commands.ranges.pos_1 = [1.,1.]
    env_cfg.commands.ranges.pos_2 = [-1.,-1.]
    goal2orig = torch.tensor([-1., 1.]).to('cuda')
    env_cfg.commands.ranges.heading = [0.0,0.0]
    env_cfg.asset.terminate_after_contacts_on = ["base", 'FL_thigh', "FL_calf", "FR_thigh", "FR_calf"]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    def new_check_termination():  # happens before update of obs and also the relpos, filter collisions falsely detected
        env.reset_buf = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
        hor_footforce = env.contact_forces[:, env.feet_indices[:2],0:2].norm(dim=-1)
        ver_footforce = torch.abs(env.contact_forces[:, env.feet_indices[:2],2])
        foot_hor_col = torch.any(hor_footforce > 2 * ver_footforce + 10.0, dim=-1)
        env.reset_buf |= foot_hor_col
        minobjdist = torch.cat([env.obj_relpos[_obj][:].norm(dim=-1).unsqueeze(-1) for _obj in range(env.cfg.asset.object_num)], dim=-1)
        _near_obj = torch.any(minobjdist<0.95, dim=-1)  
        _near_obj = torch.logical_and(_near_obj, env.base_lin_vel[:,:2].norm(dim=-1) > 0.5)  
        _near_obj = torch.logical_or(_near_obj, torch.norm(env.contact_forces[:, 0, :], dim=-1) > 1.) 
        env.reset_buf = torch.logical_and(env.reset_buf, _near_obj)   # filter the weird collisions from simulator that cannot be explained
        env.time_out_buf = (env.timer_left <= 0) # no terminal reward for time-outs
        env.reset_buf |= env.time_out_buf
        env.reset_buf |= torch.norm(env.position_targets[:, :2] - env.root_states[:, :2], dim=1) < 0.3
    env.check_termination = new_check_termination
    env.debug_viz = True
    env.terrain_levels[:] = 9
    env.do_reset = False
    env.force_obstacle_pos = True

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print('\nLoaded policy from: {}\n'.format(task_registry.loaded_policy_path))

    # collect trajectory
    traj = {
        'pos': torch.empty((traj_len, args.num_envs, 2), dtype=torch.float32),
        'ra_obs': torch.empty((traj_len, args.num_envs, 5), dtype=torch.float32),
    }
    obs = env.get_observations()
    for i in range(traj_len):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)
        traj['pos'][i] = env.root_states[:, :2] - (env.position_targets[:, :2] + goal2orig)

        v = env.base_lin_vel[:, :1]  # robot longitudinal velocity
        goal_xy = obs[:, 10:12]  # goal relative xy position
        obst_xy = env.obj_relpos[0][:, :2]  # obstacle relative xy position
        traj['ra_obs'][i] = torch.cat((v, goal_xy, obst_xy), dim=1)

    for k, v in traj.items():
        traj[k] = v.numpy()

    # save trajectory
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough', 'visualize')
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, 'traj.npz'), **traj)


def visualize_trajectory(repaired: bool = False):
    # load trajectory
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough', 'visualize')
    traj = np.load(os.path.join(path, 'traj.npz'))

    # filter infeasible trajectories
    pos = traj['pos']
    ra_obs = traj['ra_obs']
    value = load_value_function(repaired=False)
    with torch.no_grad():
        v = value(torch.from_numpy(ra_obs[0]).to('cuda')).squeeze(1).cpu().numpy()
    dist = np.linalg.norm(pos, axis=2)
    infe = (v <= 0) & np.any(dist < 0.4, axis=0)
    print("Infeasible trajectory num:", np.sum(infe))
    pos = pos[:, infe]
    ra_obs = ra_obs[:, infe]

    # downsampling
    pos = pos[:, :min(pos.shape[1], 10)]
    ra_obs = ra_obs[:, :min(ra_obs.shape[1], 10)]
    if repaired:
        value = load_value_function(repaired=True)
    with torch.no_grad():
        v = value(torch.from_numpy(ra_obs).to('cuda')).squeeze(2).cpu().numpy()

    # visualize trajectory
    fig, ax = plt.subplots()

    obst = Circle((0, 0), 0.4, color='r', alpha=0.3)
    ax.add_patch(obst)
    goal = Circle((1, -1), 0.6, color='g', alpha=0.3)
    ax.add_patch(goal)

    dist = np.linalg.norm(pos, axis=2)
    pre_pos = []
    pre_v = []
    v_min, v_max = np.inf, -np.inf
    for i in range(pos.shape[1]):
        t = np.argmax(dist[:, i] < 0.4) + 2
        pre_pos.append(pos[:t, i])
        pre_v.append(v[:t, i])
        v_min = min(v_min, v[:t, i].min())
        v_max = max(v_max, v[:t, i].max())
    v_min -= (v_max - v_min) * 0.01
    v_max += (v_max - v_min) * 0.01
    print(f'v_min: {v_min}, v_max: {v_max}')

    for pp, pv in zip(pre_pos, pre_v):
        s = ax.scatter(pp[:, 0], pp[:, 1], c=pv, vmin=v_min, vmax=v_max)
    cb = fig.colorbar(s, ax=ax)

    for i in range(pos.shape[1]):
        ax.add_patch(Circle(pos[0, i], 0.022, color='r', fill=False))
        cb.ax.plot([0, 1], [v[0, i], v[0, i]], c='r')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

    # save figure
    if repaired:
        plt.savefig(os.path.join(path, f'trajectory_repaired.png'), dpi=300)
    else:
        plt.savefig(os.path.join(path, f'trajectory.png'), dpi=300)
    plt.close()


def load_value_function(repaired: bool = False):
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough', 'exported')
    if repaired:
        onnx_model = onnx.load(os.path.join(path, 'value_repair.onnx'))
        value = ConvertModel(onnx_model, experimental=True)
        value.to('cuda')
    else:
        value = torch.load(os.path.join(path, '05_13_20-29-06_model_4000_ra.pt'))
    return value


def visualize_value_function(repaired: bool = False):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xm, ym = np.meshgrid(x, y)
    th = np.pi / 4
    rot_mat = np.stack((
        np.array((np.cos(th), -np.sin(th))),
        np.array((np.sin(th), np.cos(th)))),
    )
    goal_xy = (rot_mat @ np.stack((1 - xm, -1 - ym), axis=2)[..., np.newaxis])[..., 0]
    obst_xy = (rot_mat @ np.stack((-xm, -ym), axis=2)[..., np.newaxis])[..., 0]
    ra_obs = np.concatenate((
        np.ones_like(xm)[..., np.newaxis],
        goal_xy,
        obst_xy,
    ), axis=2)

    value = load_value_function(repaired=repaired)
    with torch.no_grad():
        vm = value(torch.from_numpy(ra_obs).float().to('cuda')).squeeze(2).cpu().numpy()

    fig, ax = plt.subplots()
    cf = ax.contourf(xm, ym, vm)
    obst = Circle((0, 0), 0.4, color='k', fill=False)
    ax.add_patch(obst)
    goal = Circle((1, -1), 0.6, color='k', fill=False)
    ax.add_patch(goal)
    zero = ax.contour(cf, levels=[0], colors='r')
    cb = fig.colorbar(cf)
    cb.add_lines(zero)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough', 'visualize')
    os.makedirs(path, exist_ok=True)
    if repaired:
        plt.savefig(os.path.join(path, f'heatmap_repaired.png'), dpi=300)
    else:
        plt.savefig(os.path.join(path, f'heatmap.png'), dpi=300)
    plt.close()


def visualize_obs_trajectory():
    # load trajectory
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough')
    traj = np.load(os.path.join(path, 'visualize', 'traj.npz'))
    env_obs = traj['ra_obs'][:, 1]

    done = (np.linalg.norm(env_obs[:, 1:3], axis=1) < 0.4) | (np.linalg.norm(env_obs[:, 3:5], axis=1) < 0.3)
    env_obs = env_obs[:np.argmax(done)]

    model = torch.load(os.path.join(path, 'exported', '05_13_20-29-06_model_4000_nndm.pt'))
    model_obs = [torch.from_numpy(env_obs[0]).to('cuda')]
    with torch.no_grad():
        for _ in range(env_obs.shape[0] - 1):
            model_obs.append(model_obs[-1] + model(model_obs[-1]))
    model_obs = torch.stack(model_obs).cpu().numpy()

    for i in range(env_obs.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(env_obs[:, i], label='env')
        ax.plot(model_obs[:, i], label='model')
        ax.legend()
        ax.set_xlabel('Time step')
        ax.set_ylabel(f'Obs_{i}')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'visualize', f'obs_traj_{i}.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    # observation_distribution()

    # args = get_args()
    # args.task = 'go1_pos_rough'
    # args.load_run = '05_13_20-29-06_'
    # args.num_envs = 1000
    # args.headless = True
    # args.seed = 1
    # collect_trajectory(args)

    visualize_trajectory(repaired=False)

    # visualize_value_function(repaired=True)

    # visualize_obs_trajectory()
