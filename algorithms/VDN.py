import os
import imageio
import random

import torch
from .DQN import DQNAgent4VDN

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class VDNNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_values:list[torch.Tensor]):
        return torch.stack(q_values, dim=0).sum(dim=0)

class VDNAgent:
    """
    这里VDN与IQL不同，IQL可以逐个智能体行动并更新，VDN只能所有智能体一起行动一轮并同时更新。
    """
    def __init__(self, num_agents, state_dim, action_dim, buffer_size=10000, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, device='cpu'):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.device = device
        self.agents = [DQNAgent4VDN(state_dim, action_dim, buffer_size, lr, gamma, epsilon, epsilon_decay,
                                epsilon_min, batch_size, device) for _ in range(num_agents)]
        self.vdn_net = VDNNet()  # 无网络，用torch加和保证梯度

        self.criterion = nn.MSELoss()

    def select_actions(self, observation, agent_idx):
        return self.agents[agent_idx].select_action(observation)

    def update(self):
        # 所有agent一起update
        sample_random_seed = random.randint(0, 10000)
        if len(self.agents[0].replay_buffer) < self.batch_size:
            return

        current_q_s, target_q_s = [], []
        for agent in self.agents:
            current_q, target_q = agent.calculate_q(sample_random_seed=sample_random_seed)
            current_q_s.append(current_q)
            target_q_s.append(target_q)

        current_q_total = self.vdn_net(current_q_s)
        target_q_total = self.vdn_net(target_q_s)

        loss = self.criterion(current_q_total, target_q_total)
        for agent in self.agents:
            agent.opt_zero_grad()
        loss.backward()
        for agent in self.agents:
            agent.opt_step()

    def update_target_networks(self):
        for agent in self.agents:
            agent.update_target_network()

    def lr_step(self):
        for agent in self.agents:
            agent.lr_step()

    def update_epsilon(self):
        for agent in self.agents:
            agent.epsilon_update()

    def add_experience(self, agent_index, observation, action, reward, next_state, done):
        # 将经验添加到对应智能体的replay buffer中
        self.agents[agent_index].replay_buffer.add(observation, action, reward, next_state, done)

    def save_model(self, path):
        # 创建一个包含所有agent模型参数的字典
        model_state_dict = {
            f'agent_{i}': agent.policy_net.state_dict()
            for i, agent in enumerate(self.agents)
        }
        torch.save(model_state_dict, path)

    def load_model(self, path):
        model_state_dict = torch.load(path, map_location=self.device)
        # 将模型参数加载到每个agent
        for i, agent in enumerate(self.agents):
            agent.policy_net.load_state_dict(model_state_dict[f'agent_{i}'])


def train_vdn(env, vdn, num_episodes, num_agents, seed, env_name='default', save_path=None, load_path=None, fig_path=None):
    if load_path:
        print('loading trained model..\n')
        vdn.load_model(load_path)
    else:
        print('training from scratch\n')

    total_step = 0
    episode_rewards = []

    for episode in range(num_episodes):
        env.reset(seed=seed)
        episode_reward = 0
        update_counter = 0

        # 把所有智能体的一次交互存储，一起更新
        agent_idxes = []
        observations = []
        actions = []
        next_rewards = []
        next_observations = []
        terminations = []

        for agent in env.agent_iter():
            update_counter += 1
            total_step += 1

            # 获取当前agent和观测
            agent_idx = int(agent.split('_')[-1])
            observation, reward, termination, truncation, info = env.last()
            observation = observation.flatten()

            # 选择动作并执行
            if termination or truncation:
                action = None
            else:
                action = vdn.select_actions(observation, agent_idx)
            env.step(action)

            episode_reward += reward

            # 添加经验
            next_observation, next_reward, termination, truncation, info = env.last()
            next_observation = next_observation.flatten()
            agent_idxes.append(agent_idx)
            observations.append(observation)
            actions.append(action)
            next_rewards.append(next_reward)
            next_observations.append(next_observation)
            terminations.append(termination)

            if update_counter % num_agents == 0:  # 所有agent都执行完一次操作，集中更新经验池和训练
                for i, agent_idx in enumerate(agent_idxes):
                    vdn.add_experience(agent_idx, observations[i], actions[i],
                                       next_rewards[i], next_observations[i], terminations[i])

                vdn.update()
                if total_step % 100 == 0:
                    vdn.update_target_networks()
                    total_step = 0

            if termination or truncation:
                break

        vdn.lr_step()
        vdn.update_epsilon()

        print(f'Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}')
        episode_rewards.append(episode_reward)

    # 保存训练好的模型
    if save_path:
        vdn.save_model(save_path)
        print('model saved at', save_path)

    # 画reward变化的折线图
    if fig_path:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.title('Rewards in: ' + env_name)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig(os.path.join(fig_path, env_name + '.png'))
        print('rewards fig saved at', os.path.join(fig_path, env_name + '.png'))

    env.close()
    return episode_rewards

def visualize_vdn(env, vdn, seed, env_name='default', load_path=None, video_path=None):
    if video_path is None:
        raise FileNotFoundError('video_path cannot be None')

    if load_path is None:
        print('warning: no trained model, exhibiting random case.\n')
    else:
        print(f'loading trained model from {load_path}\n')
        vdn.load_model(load_path)

    frames = []
    env.reset(seed=seed)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if 'pursuit' in env_name:
            frames.append(env.env.render())  # 对于pursuit的源码，这样调用可以获得全局观测。
        elif 'pong' in env_name:
            frames.append(observation)   # fixme 这里展示的视频全是纯黑图片。
        elif 'connect' in env_name:
            frames.append(env.state())   # connect4
        else:
            pass

        if termination or truncation:
            action = None
        else:
            action = vdn.select_actions(observation.flatten(), int(agent.split('_')[-1]))

        env.step(action)
    env.close()

    imageio.mimsave(os.path.join(video_path, env_name + '.mp4'), frames)
    print('game video saved at', os.path.join(video_path, env_name + '.mp4'))

