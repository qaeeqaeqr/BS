from pettingzoo.sisl import pursuit_v4
from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.classic import connect_four_v3

from configs import alg_config, env_config

from algorithms.IQL import visualize_iql, IQL
from algorithms.CTD_IQL import visualize_ctdiql, CTDIQL
from algorithms.VDN import visualize_vdn, VDNAgent
from algorithms.CTD_VDN import visualize_ctdvdn, CTDVDNAgent

from utils import *
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

pursuit = env_config.PersuitEnvConfig()
pong = env_config.PongEnvConfig()
connect4 = env_config.Connect4EnvConfig()

pursuit_env = pursuit_v4.env(
    max_cycles=pursuit.max_cycles,
    x_size=pursuit.x_size,
    y_size=pursuit.y_size,
    shared_reward=pursuit.shared_reward,
    n_evaders=pursuit.n_evaders,
    n_pursuers=pursuit.n_pursuers,
    obs_range=pursuit.obs_range,
    n_catch=pursuit.n_catch,
    freeze_evaders=pursuit.freeze_evaders,
    tag_reward=pursuit.tag_reward,
    catch_reward=pursuit.catch_reward,
    urgency_reward=pursuit.urgency_reward,
    surround=pursuit.surround,
    constraint_window=pursuit.constraint_window,
    render_mode=pursuit.render_mode,
)

pong_env = cooperative_pong_v5.env(
    ball_speed=pong.ball_speed,
    left_paddle_speed=pong.left_paddle_speed,
    right_paddle_speed=pong.right_paddle_speed,
    cake_paddle=pong.cake_paddle,
    max_cycles=pong.max_cycles,
    bounce_randomness=pong.bounce_randomness,
    max_reward=pong.max_reward,
    off_screen_penalty=pong.off_screen_penalty,
    render_mode=pong.render_mode,
)

connect4_env = connect_four_v3.env(
    render_mode=connect4.render_mode
)


def visualize_IQL_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_iql = IQL(num_agents=pursuit.n_pursuers,
                      state_dim=(pursuit.obs_range ** 2) * 3,
                      action_dim=pursuit.n_actions,
                      buffer_size=alg_config.buffer_size,
                      lr=alg_config.lr,
                      gamma=alg_config.gamma,
                      epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                      epsilon_decay=alg_config.epsilon_decay,
                      epsilon_min=alg_config.epsilon_min,
                      batch_size=alg_config.batch_size,
                      device=alg_config.device, )
    visualize_iql(testing_env, testing_iql,
                  seed=alg_config.test_seed,
                  env_name='visualize IQL in pursuit',
                  load_path=alg_config.IQL_persuit_model_path,
                  video_path=alg_config.output_dir)


def visualize_IQL_on_pong():
    testing_env = cooperative_pong_v5.env(
        ball_speed=pong.ball_speed,
        left_paddle_speed=pong.left_paddle_speed,
        right_paddle_speed=pong.right_paddle_speed,
        cake_paddle=pong.cake_paddle,
        max_cycles=pong.max_cycles,
        bounce_randomness=pong.bounce_randomness,
        max_reward=pong.max_reward,
        off_screen_penalty=pong.off_screen_penalty,
        render_mode='human',
        render_fps=35,
    )

    testing_iql = IQL(num_agents=2,
                      state_dim=48*28*2,
                      action_dim=pong.n_actions,
                      buffer_size=alg_config.buffer_size,
                      lr=alg_config.lr,
                      gamma=alg_config.gamma,
                      epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                      epsilon_decay=alg_config.epsilon_decay,
                      epsilon_min=alg_config.epsilon_min,
                      batch_size=alg_config.batch_size,
                      device=alg_config.device, )

    visualize_iql(testing_env, testing_iql,
                  seed=alg_config.test_seed,
                  env_name='visualize IQL in pong',
                  load_path=alg_config.IQL_pong_model_path,
                  video_path=alg_config.output_dir)


def visualize_CTDIQL_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_ctdiql = CTDIQL(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,
                            zeta=alg_config.zeta,
                            lr_var=alg_config.lr_var, )
    visualize_ctdiql(testing_env, testing_ctdiql,
                     seed=alg_config.test_seed,
                     env_name='visualize IQL in pursuit',
                     load_path=alg_config.IQL_persuit_model_path,
                     video_path=alg_config.output_dir)

def visualize_VDN_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_vdn = VDNAgent(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,)

    visualize_vdn(testing_env, testing_vdn,
                  seed=alg_config.test_seed,
                  env_name='visualize VDN in pursuit',
                  load_path=alg_config.VDN_persuit_model_path,
                  video_path=alg_config.output_dir)

def visualize_CTDVDN_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_ctdvdn = CTDVDNAgent(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,
                            zeta=alg_config.zeta,
                            lr_var=alg_config.lr_var, )

    visualize_ctdvdn(testing_env, testing_ctdvdn,
                     seed=alg_config.test_seed,
                     env_name='visualize CTD_VDN in pursuit',
                     load_path=alg_config.CTDVDN_persuit_model_path,
                     video_path=alg_config.output_dir)


def plot_rewards(file_path):
    assert file_path.endswith('.pkl')
    with open(file_path, 'rb') as f:
        rewards = pickle.load(f)
        rewards[:] = rewards[::1000]

    plt.plot(rewards)
    folder = file_path.split('/')
    prefix = folder.pop()
    folder = '/'.join(folder)
    plt.savefig(folder + '/' + prefix + '.png')


def shade_plot(file_paths: str, interval=1):
    '''

    :param file_paths: e.g.: '../outputs/iql-reward-'
    :return:
    '''
    folder = file_paths.split('/')
    prefix = folder.pop()
    folder = '/'.join(folder)

    target_files = []

    for file_name in os.listdir(folder):
        if file_name.endswith('.pkl') and file_name.startswith(prefix):
            target_files.append(folder + '/' + file_name)

    target_files, std = _select(target_files)
    all_rewards = []

    # 读取pkl文件
    for file_path in target_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        all_rewards.append(data[::interval])  # [::2]统一画图横轴长度

    # 假设所有文件中列表的长度是相同的
    episode_lengths = len(all_rewards[0])

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # 绘制阴影折线图
    plt.figure(figsize=(10, 6))

    plt.plot(mean_rewards)
    plt.fill_between(range(episode_lengths), mean_rewards - std_rewards, mean_rewards + std_rewards,
                         alpha=0.3)

    plt.xlabel('Episode (*1000)')
    plt.ylabel('Reward')
    plt.ylim(-200, 150)
    plt.title(f'Rewards over Episodes with Standard Deviation Shaded (std:{round(std, 4)})')

    plt.savefig(f'{folder}/{prefix}_shaded.jpg', dpi=500)

def _select(file_paths, select_type='max', file_count=5):
    """
    从所有文件中选择5个文件的组合，计算每组的标准差，选择标准差最大的一组。

    :param file_paths: 文件路径列表
    :return: 选择的标准差最大的一组文件路径
    """

    # 生成所有可能的5个文件的组合
    all_combinations = list(combinations(file_paths, file_count))

    min_std = 1e9
    max_std = -1
    best_group = None

    # 遍历所有组合，计算每组的标准差
    for group in all_combinations:
        all_rewards = []
        for file_path in group:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            all_rewards.append(data)

        # 计算当前组合的平均奖励和标准差
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        # 如果当前组合的标准差更大，则更新最佳组合
        if select_type == 'min':
            if np.mean(std_rewards) < min_std:
                min_std = np.mean(std_rewards)
                best_group = group
        elif select_type == 'max':
            if np.mean(std_rewards) > max_std:
                max_std = np.mean(std_rewards)
                best_group = group

    if select_type == 'min':
        return best_group, min_std
    else:
        return best_group, max_std

def print_final_reward(file_paths):
    folder = file_paths.split('/')
    prefix = folder.pop()
    folder = '/'.join(folder)

    target_files = []

    for file_name in os.listdir(folder):
        if file_name.endswith('.pkl') and file_name.startswith(prefix):
            target_files.append(folder + '/' + file_name)

    target_files, std = _select(target_files)
    reward = 0

    # 读取pkl文件
    for file_path in target_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        reward += data[-1]

    print(reward / 5)


if __name__ == '__main__':
    plot_rewards('./outputs/IQL_spread2025-3-16 22:25:36.pkl')
