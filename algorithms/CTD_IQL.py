import os
import imageio

import torch
from .CTD_DQN import CTDDQNAgent

import matplotlib.pyplot as plt


class CTDIQL:
    def __init__(
            self, num_agents, state_dim, action_dim, buffer_size=10000, lr=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, device='cpu', zeta=0.1, lr_var=1e-5
        ):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.device = device

        self.agents = [
            CTDDQNAgent(state_dim=state_dim,
                        action_dim=action_dim,
                        buffer_size=buffer_size,
                        lr=lr,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min,
                        batch_size=batch_size,
                        device=device,
                        zeta=zeta,
                        lr_var=lr_var)
            for _ in range(num_agents)
        ]

    def select_actions(self, observation, agent_idx):
        return self.agents[agent_idx].select_action(observation)

    def update(self, agent_idx):
        self.agents[agent_idx].update()

    def update_target_networks(self):
        for agent in self.agents:
            agent.update_target_network()

    def add_experience(self, agent_index, observation, action, reward, next_state, done):
        # 将经验添加到对应智能体的replay buffer中
        self.agents[agent_index].replay_buffer.add(observation, action, reward, next_state, done)

    def save_model(self, path):
        # 创建一个包含所有agent模型参数的字典
        model_state_dict = {}
        for i, agent in enumerate(self.agents):
            model_state_dict[f'agent_{i}_mean'] = agent.policy_net_mean.state_dict()
            model_state_dict[f'agent_{i}_var'] = agent.policy_net_var.state_dict()
        torch.save(model_state_dict, path)

    def load_model(self, path):
        model_state_dict = torch.load(path, map_location=self.device)
        # 将模型参数加载到每个agent
        for i, agent in enumerate(self.agents):
            agent.policy_net_mean.load_state_dict(model_state_dict[f'agent_{i}_mean'])
            agent.policy_net_var.load_state_dict(model_state_dict[f'agent_{i}_var'])


def train_ctdiql(env, ctdiql, num_episodes, seed, env_name='default', save_path=None, load_path=None, fig_path=None):
    if load_path:
        print('loading trained model..\n')
        ctdiql.load_model(load_path)
    else:
        print('training from scratch\n')

    total_step = 0
    episode_rewards = []
    for episode in range(num_episodes):
        env.reset(seed=seed)
        episode_reward = 0

        actions = []
        for agent in env.agent_iter():
            agent_idx = int(agent.split('_')[-1])  # 根据环境信息获取agent编号
            total_step += 1
            observation, reward, termination, truncation, info = env.last()
            observation = observation.flatten()  # 将图像或二位数据都转成一维

            # 选择动作
            if termination or truncation:
                action = None
            else:
                action = ctdiql.select_actions(observation, int(agent.split('_')[-1]))
            actions.append(action)
            episode_reward += reward

            # 执行动作
            env.step(action)

            # 将经验添加到replay buffer中
            next_observation, next_reward, termination, truncation, info = env.last()
            next_observation = next_observation.flatten()
            ctdiql.add_experience(agent_idx,
                               observation, action, next_reward, next_observation, termination)

            # 从经验池采样训练智能体
            ctdiql.update(agent_idx)

            # 每100步更新一次目标网络
            if total_step % 100 == 0:
                ctdiql.update_target_networks()
                total_step = 0

            # 检查是否达到终止条件
            if termination or truncation:
                break

        print(f'Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}')
        episode_rewards.append(episode_reward)

    # 保存训练好的模型
    if save_path:
        ctdiql.save_model(save_path)
        print('model saved at', save_path)

    # 画reward变化的折线图
    if fig_path:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, marker='o')
        plt.title('Rewards in: ' + env_name)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig(os.path.join(fig_path, env_name + '.png'))
        print('rewards fig saved at', os.path.join(fig_path, env_name + '.png'))

    env.close()
    return episode_rewards

def visualize_ctdiql(env, ctdiql, seed, env_name='default', load_path=None, video_path=None):
    if video_path is None:
        raise FileNotFoundError('video_path cannot be None')

    if load_path is None:
        print('warning: no trained model, exhibiting random case.\n')
    else:
        print(f'loading trained model from {load_path}\n')
        ctdiql.load_model(load_path)

    frames = []
    env.reset(seed=seed)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if 'pursuit' in env_name:
            frames.append(env.env.render())  # 对于pursuit的源码，这样调用可以获得全局观测。
        elif 'pong' in env_name:
            frames.append(observation)   # pong里面的观测是全局的
        elif 'connect' in env_name:
            frames.append(env.state())   # connect4
        else:
            pass

        if termination or truncation:
            action = None
        else:
            action = ctdiql.select_actions(observation.flatten(), int(agent.split('_')[-1]))

        env.step(action)
    env.close()

    imageio.mimsave(os.path.join(video_path, env_name + '.mp4'), frames)
    print('game video saved at', os.path.join(video_path, env_name + '.mp4'))


