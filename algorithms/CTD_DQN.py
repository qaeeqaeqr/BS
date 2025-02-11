import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class QNetMean(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetMean, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        return self.network(x)


class QNetVar(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetVar, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax()
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


class CTDDQNAgent:
    def __init__(
            self, state_dim, action_dim, buffer_size=10000, lr=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, device='cpu', zeta=0.1, lr_var=1e-5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device
        self.zeta = zeta

        self.policy_net_mean = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net_mean = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net_mean.load_state_dict(self.policy_net_mean.state_dict())
        self.target_net_mean.eval()
        self.optimizer_mean = optim.Adam(self.policy_net_mean.parameters(), lr=lr)

        self.policy_net_var = QNetVar(state_dim, action_dim).to(self.device)
        self.target_net_var = QNetVar(state_dim, action_dim).to(self.device)
        self.target_net_var.load_state_dict(self.policy_net_var.state_dict())
        self.target_net_var.eval()
        self.optimizer_var = optim.Adam(self.policy_net_var.parameters(), lr=lr_var)

        self.lr_scheduler_mean = lr_scheduler.ExponentialLR(self.optimizer_mean, gamma=0.9995, last_epoch=-1)
        self.lr_scheduler_var = lr_scheduler.ExponentialLR(self.optimizer_var, gamma=0.9995, last_epoch=-1)

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

    def select_action(self, state):
        # Epsilon-greedy 策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values_mean = self.policy_net_mean(state).detach().cpu()
                q_values_var = self.policy_net_var(state).detach().cpu()
            mean_var_linear = q_values_mean - self.zeta * q_values_var
            return torch.argmax(mean_var_linear).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验回放池中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # mean var网络更新公式的实现。
        current_q_mean = self.policy_net_mean(states).gather(1, actions)
        target_q_mean_reward = rewards + (1 - dones) * self.gamma * self.target_net_mean(next_states).max(1, keepdim=True)[0]
        current_q_var = self.policy_net_var(states).gather(1, actions)
        target_q_var = (1 - dones) * self.gamma * self.target_net_var(next_states).max(1, keepdim=True)[0]

        # loss及bp
        loss_mean = self.criterion(current_q_mean, target_q_mean_reward)
        loss_var = self.criterion(current_q_var, (target_q_mean_reward - current_q_mean)**2 + target_q_var)
        loss_total = loss_mean + loss_var  # 由于pytorch只保存一个计算图，所以把两个loss加起来，只做一次bp
        self.optimizer_mean.zero_grad()
        self.optimizer_var.zero_grad()
        loss_total.backward()
        self.optimizer_mean.step()
        self.optimizer_var.step()

    def epsilon_update(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net_mean.load_state_dict(self.policy_net_mean.state_dict())
        self.target_net_var.load_state_dict(self.policy_net_var.state_dict())

    def lr_step(self):
        # self.lr_scheduler_mean.step()
        # self.lr_scheduler_var.step()
        pass

class CTDDQNAgent4VDN:
    def __init__(
            self, state_dim, action_dim, buffer_size=10000, lr=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, device='cpu', zeta=0.1, lr_var=1e-5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device
        self.zeta = zeta

        self.policy_net_mean = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net_mean = QNetMean(state_dim, action_dim).to(self.device)
        self.target_net_mean.load_state_dict(self.policy_net_mean.state_dict())
        self.target_net_mean.eval()
        self.optimizer_mean = optim.Adam(self.policy_net_mean.parameters(), lr=lr)

        self.policy_net_var = QNetVar(state_dim, action_dim).to(self.device)
        self.target_net_var = QNetVar(state_dim, action_dim).to(self.device)
        self.target_net_var.load_state_dict(self.policy_net_var.state_dict())
        self.target_net_var.eval()
        self.optimizer_var = optim.Adam(self.policy_net_var.parameters(), lr=lr_var)

        self.lr_scheduler_mean = lr_scheduler.ExponentialLR(self.optimizer_mean, gamma=0.9995, last_epoch=-1)
        self.lr_scheduler_var = lr_scheduler.ExponentialLR(self.optimizer_var, gamma=0.9995, last_epoch=-1)

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

    def select_action(self, state):
        # Epsilon-greedy 策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values_mean = self.policy_net_mean(state).detach().cpu()
                q_values_var = self.policy_net_var(state).detach().cpu()
            mean_var_linear = q_values_mean - self.zeta * q_values_var
            return torch.argmax(mean_var_linear).item()

    def _sample_buffer(self, buffer, k, random_seed=None):
        # 创建一个随机数生成器实例，并指定随机种子
        random_gen = random.Random()
        if random_seed is not None:
            random_gen.seed(random_seed)

        deq_list = list(buffer.buffer)
        sample = random_gen.sample(deq_list, k)

        return sample

    def calculate_q(self, sample_random_seed=42):
        """计算Q均值和方差"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None, None

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

        # mean var网络更新公式的实现。
        current_q_mean = self.policy_net_mean(states).gather(1, actions)
        target_q_mean = rewards + (1 - dones) * self.gamma * \
                               self.target_net_mean(next_states).max(1, keepdim=True)[0]
        current_q_var = self.policy_net_var(states).gather(1, actions)
        target_q_var = (target_q_mean - current_q_mean)**2 + (
                (1 - dones) * self.gamma * self.target_net_var(next_states).max(1, keepdim=True)[0])

        return current_q_mean, target_q_mean, current_q_var, target_q_var

    def opt_zero_grad(self):
        self.optimizer_mean.zero_grad()
        self.optimizer_var.zero_grad()

    def opt_step(self):
        self.optimizer_mean.step()
        self.optimizer_var.step()

    def epsilon_update(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def lr_step(self):
        # self.lr_scheduler_mean.step()
        # self.lr_scheduler_var.step()
        pass

    def update_target_network(self):
        self.target_net_mean.load_state_dict(self.policy_net_mean.state_dict())
        self.target_net_var.load_state_dict(self.policy_net_var.state_dict())


