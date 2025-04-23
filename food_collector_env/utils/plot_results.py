import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
from itertools import combinations

# Function that calculates moving average of a series
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# Function that plots scores and epsilon against number of episodes
def plot_scores_epsilon(rewards_history, epsilon_history, moving_avg_window=100):
    
    team_rewards = [sum(x) for x in zip(*rewards_history)]
    
    f, axarr = plt.subplots(1,1, figsize=(10,6))
    for i in range(len(rewards_history)):
        axarr.plot(movingaverage(np.array(rewards_history[i]), moving_avg_window), label=f'Agent {i+1} (MA)')
    axarr.plot(movingaverage(np.array(team_rewards), moving_avg_window), label=f'Team rewards (MA)')
    axarr.set_xlabel('Episodes')
    axarr.set_ylabel('Scores')
    axarr.legend()

    plt.savefig('scores.png')
    plt.show()


def plot_rewards(file_path):
    assert file_path.endswith('.pkl')
    with open(file_path, 'rb') as f:
        rewards = pickle.load(f)

    plt.plot(rewards)
    folder = file_path.split('/')
    prefix = folder.pop()
    folder = '/'.join(folder)
    plt.savefig(folder + '/' + prefix + '.png')


def shade_plot(file_paths: str, select_type='min', interval=1, color='#48C0AA'):
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

    target_files, std = _select(target_files, select_type)
    all_rewards = []

    # 读取pkl文件
    for file_path in target_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        all_rewards.append(data[::interval])

    # 假设所有文件中列表的长度是相同的
    episode_lengths = len(all_rewards[0])

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # 绘制阴影折线图
    plt.figure(figsize=(10, 6))

    plt.plot(mean_rewards, color=color)
    plt.fill_between(range(episode_lengths), mean_rewards - std_rewards, mean_rewards + std_rewards, color=color,
                         alpha=0.3)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-200, 150)
    print(f'std:{round(std, 4)}')

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

def print_final_reward(file_paths, select_type='min'):
    folder = file_paths.split('/')
    prefix = folder.pop()
    folder = '/'.join(folder)

    target_files = []

    for file_name in os.listdir(folder):
        if file_name.endswith('.pkl') and file_name.startswith(prefix):
            target_files.append(folder + '/' + file_name)

    target_files, std = _select(target_files, select_type)
    reward = 0

    # 读取pkl文件
    for file_path in target_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        reward += data[-1]

    print(reward / 5)


if __name__ == '__main__':
    # color='#48C0AA' color='#EF767A'
    file_paths = '../outputs/ctdvdn_reward'
    select_type = 'min'
    shade_plot(file_paths, select_type=select_type, interval=100, color='#EF767A')
    print_final_reward(file_paths, select_type='max')

