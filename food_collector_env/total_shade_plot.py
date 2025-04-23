import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from itertools import combinations
import logging

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False


def average_every_k(numbers, k=100):
    """
    将数据按每 k 个点分组，并计算每组的平均值。
    """
    result = []
    n = len(numbers)
    for i in range(0, n, k):
        chunk = numbers[i:i + k]
        if chunk:
            avg = sum(chunk) / len(chunk)
            result.append(avg)
    return result


def shade_plot_multiple_algorithms(algorithm_data, select_type, interval=100, file_count=5,
                                   x_label='Episode', y_label='Reward', title='Algorithm Comparison', labelsize=16, ticksize=13,
                                   ylim=None, save_path=None):
    """
    绘制多个算法的阴影折线图对比。

    :param algorithm_data: 字典，键为算法名称，值为文件路径前缀（e.g., '../outputs/iql-reward-'）
    :param select_type: 选择标准差最大或最小的组合 ('min' 或 'max')
    :param interval: 数据分组间隔
    :param file_count: 每个算法选择的文件数量
    :param x_label: x 轴标签
    :param y_label: y 轴标签
    :param title: 图表标题
    :param ylim: y 轴范围 (tuple, e.g., (-0.5, 2.5))
    :param save_path: 图表保存路径
    """
    plt.figure(figsize=(12, 12))

    # 遍历每个算法
    count = -1
    color = ['#48C0AA', '#EF767A']  # ['#60966d', '#5b3660', '#ffc839', '#e90f44', '#63adee']
    for algorithm_name, file_prefix in algorithm_data.items():
        count += 1
        # 验证文件夹路径
        folder = os.path.dirname(file_prefix)
        prefix = os.path.basename(file_prefix)

        if not os.path.exists(folder):
            logging.error(f"文件夹路径不存在: {folder}")
            continue

        target_files = []
        for file_name in os.listdir(folder):
            if file_name.endswith('.pkl') and file_name.startswith(prefix):
                target_files.append(os.path.join(folder, file_name))

        if not target_files:
            logging.error(f"未找到匹配的文件: {file_prefix}")
            continue

        # 选择最佳文件组合
        selected_files, std = _select(target_files, select_type=select_type[count], file_count=file_count)
        if not selected_files:
            logging.error(f"未找到合适的文件组合: {algorithm_name}")
            continue

        all_rewards = []
        for file_path in selected_files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if count == 0:
                    averaged_data = average_every_k(data, 2)
                else:
                    averaged_data = average_every_k(data, 2)
                all_rewards.append(np.array(averaged_data))
            except Exception as e:
                logging.error(f"读取文件 {file_path} 时出错: {e}")
                continue

        # 检查所有文件的数据长度是否一致
        if not all_rewards:
            logging.error(f"未找到有效的数据: {algorithm_name}")
            continue

        # 找到所有数据的最小长度
        min_length = min(len(rewards) for rewards in all_rewards)
        all_rewards = [rewards[:min_length] for rewards in all_rewards]

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        smoothed_mean_rewards = gaussian_filter1d(mean_rewards, sigma=2)
        smoothed_std_rewards = gaussian_filter1d(std_rewards, sigma=2) / 2

        # 绘制当前算法的阴影折线图
        plt.plot(smoothed_mean_rewards, color=color[count], label=algorithm_name)
        plt.fill_between(range(len(smoothed_mean_rewards)), smoothed_mean_rewards - smoothed_std_rewards,
                         smoothed_mean_rewards + smoothed_std_rewards,
                         color=color[count], alpha=0.3)

    # 设置图表属性
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    if ylim:
        plt.ylim(ylim)
    plt.legend(fontsize=20)

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=800)
        logging.info(f"图表已保存到: {save_path}")
    else:
        plt.show()


def _select(file_paths, select_type='min', file_count=5, metric='std'):
    """
    从文件中选择最佳组合。
    """
    if len(file_paths) < file_count:
        logging.error(f"文件数量不足: 需要至少 {file_count} 个文件")
        return None, None

    all_combinations = list(combinations(file_paths, file_count))
    best_group = None
    best_metric_value = None

    for group in all_combinations:
        all_rewards = []
        for file_path in group:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                all_rewards.append(np.array(data))
            except Exception as e:
                logging.error(f"读取文件 {file_path} 时出错: {e}")
                continue

        if len(all_rewards) < file_count:
            continue

        mean_rewards = np.mean(np.array(all_rewards), axis=0)
        std_rewards = np.std(np.array(all_rewards), axis=0)

        if metric == 'std':
            current_metric = np.mean(std_rewards)
        elif metric == 'mean':
            current_metric = np.mean(mean_rewards)
        else:
            logging.error(f"不支持的指标: {metric}")
            return None, None

        if select_type == 'min':
            if best_metric_value is None or current_metric < best_metric_value:
                best_metric_value = current_metric
                best_group = group
        elif select_type == 'max':
            if best_metric_value is None or current_metric > best_metric_value:
                best_metric_value = current_metric
                best_group = group

    if best_group is None:
        logging.error("未找到合适的文件组合")
        return None, None

    return list(best_group), best_metric_value



if __name__ == '__main__':
    # algorithm_data = {
    #     'zeta=0': './outputs/iql_reward',
    #     'zeta=0.001': './outputs/ctdiql_zeta0.001_reward',
    #     'zeta=0.01': './outputs/ctdiql_zeta0.01_reward',
    #     'zeta=0.1': './outputs/ctdiql_zeta0.1_reward',
    #     'zeta=1': './outputs/ctdiql_zeta1_reward',
    # }
    # algorithm_data = {
    #     'α=1e-1': './outputs/ctdiql_zeta0.1_lrvar0.1_reward',
    #     'α=1e-2': './outputs/ctdiql_zeta0.1_lrvar0.01_reward',
    #     'α=1e-3': './outputs/ctdiql_zeta0.1_lrvar0.001_reward',
    #     'α=1e-4': './outputs/ctdiql_zeta0.1_reward',
    #     'α=1e-5': './outputs/ctdiql_zeta0.1_lrvar1e-05_reward'
    # }
    # algorithm_data = {
    #     'VDN': './outputs/vdn_reward',
    #     'MA-CTD-VDN': './outputs/ctdvdn_reward',
    # }
    algorithm_data = {
        'IQL': './outputs/iql_reward',
        'MA-CTD-IQL': './outputs/ctdiql_zeta0.1_reward',
    }

    shade_plot_multiple_algorithms(
        algorithm_data,
        select_type=['max', 'min'],
        interval=200,
        file_count=5,
        x_label='训练轮数',
        y_label='奖励',
        labelsize=28,
        ticksize=22,
        ylim=(-50, 80),
        save_path='./outputs/iql_cmp.jpg'
    )
