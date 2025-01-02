"""
主要构建Q网络（用于预测均值和方差的网络架构可能不同）、经验池和训练脚本。

"""
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class QNetMean(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetMean, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
            self, state_dim, action_dim, buffer_size=10000, lr=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device

        self.policy_net = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

    def select_action(self, state):
        # Epsilon-greedy 策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验回放池中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # print(states.shape)   # batch_size*observation_dim

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        target_q = rewards + (1 - dones) * self.gamma * self.target_net(next_states).max(1, keepdim=True)[0]

        # 更新策略网络
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DQNAgent4VDN:
    def __init__(
            self, state_dim, action_dim, buffer_size=10000, lr=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device

        self.policy_net = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

    def select_action(self, state):
        # Epsilon-greedy 策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def _sample_buffer(self, buffer, k, random_seed=None):
        # 创建一个随机数生成器实例，并指定随机种子
        random_gen = random.Random()
        if random_seed is not None:
            random_gen.seed(random_seed)

        deq_list = list(buffer.buffer)
        sample = random_gen.sample(deq_list, k)

        return sample

    def calculate_q(self, sample_random_seed=42):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # 从经验回放池中采样
        batch = self._sample_buffer(self.replay_buffer, self.batch_size,
                                    random_seed=sample_random_seed)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # 计算 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        target_q = rewards + (1 - dones) * self.gamma * self.target_net(next_states).max(1, keepdim=True)[0]

        return current_q, target_q

    def opt_zero_grad(self):
        self.optimizer.zero_grad()

    def opt_step(self):
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
