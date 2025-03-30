import pickle
from datetime import datetime

import numpy as np
import random
from env import FoodCollectEnv

class IQLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, learning_rate_var=1e-4,
                 gamma=0.99, epsilon_max=0.9, epsilon_min=0.2, epsilon_decay=0.99, zeta=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.learning_rate_var = learning_rate_var
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.zeta = zeta
        # 初始化Q表格
        self.q_table = np.zeros((state_dim[0], state_dim[1], action_dim))
        self.q_table_var = np.zeros((state_dim[0], state_dim[1], action_dim))

    def choose_action(self, state):
        # ε-greedy策略选择动作
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            action = np.argmax(self.q_table[state[0], state[1], :] + zeta * np.sqrt(self.q_table_var[state[0], state[1], :]))
        return action

    def learn(self, state, action, reward, next_state):
        # 更新Q表格
        predict = self.q_table[state[0], state[1], action]
        target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
        self.q_table[state[0], state[1], action] = self.q_table[state[0], state[1], action] + self.learning_rate * (target - predict)
        predict_var = self.q_table_var[state[0], state[1], action]
        target_var = (target - predict)**2 + self.q_table_var[next_state[0], next_state[1], action]
        self.q_table_var[state[0], state[1], action] = self.q_table_var[state[0], state[1], action] + self.learning_rate_var * (
            target_var - predict_var)


    def epsilon_update(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def train_iql_agents(zeta=0.1):
    # 创建环境
    env = FoodCollectEnv()
    # 每个智能体的观测空间维度
    state_dim = (env.grid_size, env.grid_size)
    # 动作空间维度
    action_dim = 4
    # 创建两个独立的Q学习智能体
    agent1 = IQLAgent(state_dim, action_dim, zeta=zeta)
    agent2 = IQLAgent(state_dim, action_dim, zeta=zeta)
    # 训练参数
    episodes = 50000
    log_interval = 100
    # 用于记录每个智能体的累计奖励
    total_rewards = [[], []]

    for episode in range(episodes):
        done = [False, False]
        episode_rewards = [0, 0]
        step = 0
        states = env.reset()
        state1, state2 = tuple(states[0]), tuple(states[1])

        while not all(done):
            # 智能体1选择动作
            action1 = agent1.choose_action(state1)
            # 智能体2选择动作
            action2 = agent2.choose_action(state2)
            # 执行动作
            next_state, rewards, done, _ = env.step([action1, action2])
            next_state1 = tuple(next_state[0])
            next_state2 = tuple(next_state[1])
            # 智能体1学习
            agent1.learn(state1, action1, rewards[0], next_state1)
            # 智能体2学习
            agent2.learn(state2, action2, rewards[1], next_state2)
            # 更新状态
            state1 = next_state1
            state2 = next_state2
            # 累计奖励
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]
            step += 1

            if step > 100:
                break

        # 记录每个智能体的累计奖励
        total_rewards[0].append(episode_rewards[0])
        total_rewards[1].append(episode_rewards[1])

        # 打印训练进度
        if (episode + 1) % log_interval == 0:
            print(f"Episode: {episode + 1}, "
                  f"Agent1 Reward: {np.mean(total_rewards[0][-log_interval:]):.2f}, "
                  f"Agent2 Reward: {np.mean(total_rewards[1][-log_interval:]):.2f}")

    return agent1, agent2, [total_rewards[0][i] + total_rewards[1][i] for i in range(len(total_rewards[0]))]

if __name__ == "__main__":
    # 训练IQL智能体
    zeta = 0.1
    train_start_time = (str(datetime.now().year) + '-' + str(datetime.now().month) + '-' + str(datetime.now().day) +
                        ' ' + str(datetime.now().hour) + '-' + str(datetime.now().minute) + '-' + str(datetime.now().second))
    agent1, agent2, rewards = train_iql_agents(zeta=zeta)
    with open(f'./outputs/ctdiql_zeta{zeta}_reward_{train_start_time}.pkl', 'wb') as f:
        pickle.dump(rewards, f)
