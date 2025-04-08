import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

import pickle
from datetime import datetime
import imageio
from PIL import Image

from tqdm import tqdm
import dill
from env import FoodCollectEnv

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        return torch.tensor(np.array(s_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(np.array(a_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(np.array(r_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(np.array(done_lst), dtype=torch.float).view(batch_size, chunk_size, 1)

    def size(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, observation_space, action_space, zeta=0.1, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        self.zeta = zeta
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_q_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 64),
                                                                            nn.ReLU(),
                                                                            nn.Linear(64, self.hx_size),
                                                                            nn.ReLU()))
            setattr(self, 'agent_feature_q_var_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 64),
                                                                            nn.ReLU(),
                                                                            nn.Linear(64, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))
            setattr(self, 'agent_q_var_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        q_var_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
            x_q = getattr(self, 'agent_feature_q_{}'.format(agent_i))(obs[:, agent_i, :])
            x_q_var = getattr(self, 'agent_feature_q_var_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x_q = getattr(self, 'agent_gru_{}'.format(agent_i))(x_q, hidden[:, agent_i, :])
                next_hidden[agent_i] = x_q.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x_q).unsqueeze(1)
            q_var_values[agent_i] = getattr(self, 'agent_q_var_{}'.format(agent_i))(x_q_var).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(q_var_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q, q_var, hidden = self.forward(obs, hidden)
        out = q - self.zeta * torch.sqrt(q_var)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


def train(q, q_target, memory, optimizer, optimizer_var, gamma, zeta, batch_size, episode, update_iter=10, chunk_size=10, grad_clip_norm=1):
    _chunk_size = chunk_size if q.recurrent else 1
    for _ in range(update_iter):
        s, a, r, s_prime, done = memory.sample_chunk(batch_size, _chunk_size)

        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        loss = 0
        for step_i in range(_chunk_size):
            q_out, q_var_out, hidden = q(s[:, step_i, :, :], hidden)

            q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
            q_var_a = q_var_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
            sum_q = q_a.sum(dim=1, keepdims=True)
            sum_q_var = q_var_a.sum(dim=1, keepdims=True)

            max_q_prime, max_q_var_prime, target_hidden = q_target(s_prime[:, step_i, :, :], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
            max_q_var_prime = max_q_var_prime.max(dim=2)[0].squeeze(-1)

            target_q = r[:, step_i, :].sum(dim=1, keepdims=True) + gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i])
            target_q_var = torch.sqrt(torch.pow((target_q.detach() - sum_q), 2) +
                            gamma * max_q_var_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i]))

            loss += F.smooth_l1_loss(sum_q, target_q.detach()) + zeta * F.smooth_l1_loss(sum_q_var, target_q_var.detach())
            # print(round(sum_q[0][0].item(), 5), round(target_q[0][0].item(), 5), '\t',
            #       round(sum_q_var[0][0].item(), 5), round(target_q_var[0][0].item(), 5), '\t',)

            done_mask = done[:, step_i].squeeze(-1).bool()
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))

        optimizer.zero_grad()
        optimizer_var.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer_var.step()


def test(env, num_episodes, q):
    score = 0
    for episode_i in range(num_episodes):
        step_counter = 0
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            while not all(done):
                step_counter += 1
                action, hidden = q.sample_action(torch.Tensor(np.array(state)).unsqueeze(0), hidden, epsilon=0)
                next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist())
                score += sum(reward)
                state = next_state

                if step_counter > 100:
                    break
    return score / num_episodes


def train_VDN_agent(env_name, lr, lr_var, zeta, gamma, batch_size, buffer_limit, log_interval, max_episodes, max_epsilon,
                    min_epsilon, test_episodes, warm_up_steps, update_iter, chunk_size, update_target_interval,
                    recurrent):
    # create env.
    env = FoodCollectEnv()
    test_env = FoodCollectEnv()
    memory = ReplayBuffer(buffer_limit)

    # create networks
    q = QNet(env.observation_space, env.action_space, zeta, recurrent)
    q_target = QNet(env.observation_space, env.action_space, zeta, recurrent)
    q_target.load_state_dict(q.state_dict())

    q_params = [p for name, p in q.named_parameters() if 'agent_q_' in name or 'agent_feature_q_' in name or 'gru' in name]
    q_var_params = [p for name, p in q.named_parameters() if 'agent_q_var_' in name or 'agent_feature_q_var_' in name]
    optimizer = optim.Adam(q_params, lr=lr)
    optimizer_var = optim.Adam(q_var_params, lr=lr_var)

    # For performance monitoring
    n_agents = len(env.observation_space)
    episode_rewards = [[] for _ in range(n_agents)]
    epsilon_history = list()

    for episode_i in tqdm(range(max_episodes)):
        rewards_temp = [[] for _ in range(n_agents)]
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.5 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            step_counter = 0
            while True:
                step_counter += 1
                action, hidden = q.sample_action(torch.Tensor(np.array(state)).unsqueeze(0), hidden, epsilon)
                action = action[0].data.cpu().numpy().tolist()
                next_state, reward, done, info = env.step(action)
                memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))
                state = next_state

                for i in range(n_agents):
                    rewards_temp[i].append(reward[i])

                if all(done) or step_counter > 100:
                    # log rewards
                    for i in range(n_agents):
                        episode_rewards[i].append(sum(rewards_temp[i]))
                    epsilon_history.append(epsilon)
                    break

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, optimizer_var, gamma, zeta, batch_size, episode_i, update_iter, chunk_size)

        if episode_i % update_target_interval:
            for target_param, param in zip(q_target.parameters(), q.parameters()):
                target_param.data.copy_(0.5 * param.data + 0.5 * target_param.data)

        if (episode_i + 1) % log_interval == 0:
            test_score = test(test_env, test_episodes, q)
            print("#{:<10}/{} episodes, test score: {:.1f} n_buffer : {}, eps : {:.2f}"
                  .format(episode_i, max_episodes, test_score, memory.size(), epsilon))

    return q, episode_rewards, epsilon_history


def visualise_VDN(agent, n_episodes, epsilon):
    # create env.
    env = FoodCollectEnv()

    # create network -> agent already specified
    frames = []
    for episode_i in range(n_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = agent.init_hidden()
            step_counter = 0
            while not all(done):
                step_counter += 1
                action, hidden = agent.sample_action(torch.Tensor(np.array(state)).unsqueeze(0), hidden, epsilon)
                action = action[0].data.cpu().numpy().tolist()
                next_state, reward, done, info = env.step(action)
                state = next_state

                frame = env.render()
                frames.append(frame)

                if step_counter > 30:
                    break
    env.close()

    return frames

def resize_frame(frame, scale):
    # 将numpy数组转换为PIL图像
    img = Image.fromarray(frame)

    # 获取原始图像的尺寸
    width, height = frame.shape[1], frame.shape[0]

    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_img = img.resize((new_width, new_height), Image.NEAREST)

    # 将PIL图像转换回numpy数组
    resized_frame = np.array(resized_img)

    return resized_frame

if __name__ == '__main__':
    train_start_time = (str(datetime.now().year) + '-' + str(datetime.now().month) + '-' + str(datetime.now().day) +
                        ' ' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ':' + str(datetime.now().second))

    kwargs = {'env_name': 'dummy',
              'lr': 5e-4,
              'lr_var': 5e-6,
              'zeta': 1,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 20000, #50000
              'update_target_interval': 10,
              'log_interval': 500,
              'max_episodes': 25000,
              'max_epsilon': 0.9,
              'min_epsilon': 0.2,
              'test_episodes': 3,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
              'recurrent': False}
    VDNagent, reward_history, epsilon_history = train_VDN_agent(**kwargs)

    team_rewards = [sum(x) for x in zip(*reward_history)]
    with open(f'./outputs/ctdvdn_zeta{kwargs["zeta"]}_reward_{train_start_time}.pkl', 'wb') as f:
        pickle.dump(team_rewards, f)

    # frames = visualise_VDN(VDNagent, n_episodes=3, epsilon=0)
    # frames = [resize_frame(frame, scale=16) for frame in frames]
    # imageio.mimsave(f'./outputs/ctdvdn_zeta{kwargs["zeta"]}_video_{train_start_time}.mp4', frames, fps=2)



