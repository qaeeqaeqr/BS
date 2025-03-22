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
from utils.plot_results import plot_scores_epsilon
from food_collector_env.food_collector import Food_Collector_Env

# HYPERPARAMETERS
N_AGENTS = 2
NUM_EPISODES = 500000
EPS_DECAY = 0.9999
EPS_MIN = 0.2
STEP_SIZE = 0.1
STEP_SIZE_VAR = 0.001
GAMMA = 0.99
ZETA = 0.01
MAX_STEPS_DONE = 70

class QLearningAgent():

    def __init__(self, num_actions, num_states, eps_start=1.0, eps_decay=.9999, eps_min=1e-08, step_size=0.1, step_size_var=0.01, gamma=1):
        # Initialise agent
        self.num_actions = num_actions
        self.num_states = num_states
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.step_size = step_size
        self.step_size_var = step_size_var
        self.gamma = gamma
        self.rand_generator = np.random.RandomState(1)

        # Create an array for action-value estimates and initialize it to zero.
        self.state_dict = {}
        self.q = np.zeros((self.num_states, self.num_actions))  # The array of action-value estimates.
        self.q_var = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):

        # Update epsilon at the start of each episode
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        # Add state to dict if new + get index of state
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Choose action using epsilon greedy.
        current_q = self.q[state_idx, :]
        current_q_var = self.q_var[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)  # random action selection
        else:
            action = self.argmax(current_q - ZETA * np.sqrt(current_q_var))  # greedy action selection (mean-var linear combination)

        self.prev_state_idx = self.state_dict[state]
        self.prev_action = action
        return action

    def agent_step(self, reward, state):

        # Add state to dict if new + get index of state
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Choose action using epsilon greedy.
        current_q = self.q[state_idx, :]
        current_q_var = self.q_var[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q + ZETA * np.sqrt(current_q_var))

        # Update q values
        delta = reward + self.gamma * self.q[state_idx][action] - self.q[self.prev_state_idx][self.prev_action]
        self.q[self.prev_state_idx][self.prev_action] = self.q[self.prev_state_idx][self.prev_action] + self.step_size * delta
        self.q_var[self.prev_state_idx][self.prev_action] = self.q_var[self.prev_state_idx][self.prev_action] + \
                                                            self.step_size_var * (delta**2 + self.q_var[state_idx][action] -
                                                                                  self.q_var[self.prev_state_idx][self.prev_action])
        # print(self.q[self.prev_state_idx][self.prev_action], self.q_var[self.prev_state_idx][self.prev_action])

        self.prev_state_idx = self.state_dict[state]
        self.prev_action = action
        return action

    def agent_end(self, state, reward):
        # Add state to dict if new + get index
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Perform the last q value update
        self.q[self.prev_state_idx][self.prev_action] = self.q[self.prev_state_idx][self.prev_action] + self.step_size * reward
        self.q_var[self.prev_state_idx][self.prev_action] = self.q_var[self.prev_state_idx][self.prev_action] + \
                                                            self.step_size_var * reward

    # Takes step in testing environment, epsilon=0 and no updates made
    def test_step(self, state):

        # Add state to dict if new + get index of state
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Pick greedy action
        current_q = self.q[state_idx, :]
        current_q_var = self.q_var[state_idx, :]
        action = self.argmax(current_q + ZETA * np.sqrt(current_q_var))

        return action

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)


def train_QL_agents(n_agents, num_episodes, max_steps_done, eps_decay, eps_min, step_size, step_size_var, gamma):
    # initiate environment
    env = Food_Collector_Env(grid_size=[11, 11], n_agents=2)

    # get numer of states and actions from environment
    n_states = 3000000
    n_actions = env.action_space[0].n

    # Initialise agents
    agents = []
    for _ in range(n_agents):
        QL_agent = QLearningAgent(num_actions=n_actions, num_states=n_states,
                                  eps_start=1.0, eps_decay=eps_decay, eps_min=eps_min,
                                  step_size=step_size, step_size_var=step_size_var, gamma=gamma)
        agents.append(QL_agent)

    # Monitor the scores and epsilon for each episode
    episode_rewards = [[] for _ in range(n_agents)]
    epsilon_history = list()
    won_games = 0

    # for episode in num_episodes
    for episode in tqdm(range(num_episodes)):

        rewards_temp = [[] for _ in range(n_agents)]

        # get initial state and actions
        states = env.reset()
        for i in range(len(agents)):
            action = agents[i].agent_start(states[i])

        rewards = [0 for _ in range(n_agents)]
        steps = 0

        while True:
            steps += 1
            actions = []
            for i in range(n_agents):
                action = agents[i].agent_step(rewards[i], states[i])
                actions.append(action)

            next_states, rewards, done, info = env.step(actions)

            for i in range(n_agents):
                rewards_temp[i].append(rewards[i])

            if done[0]:
                won_games += 1
                for i in range(n_agents):
                    agents[i].agent_end(states[i], rewards[i])  # update q values last time

                if episode % 1000 == 0:
                    for i in range(n_agents):
                        episode_rewards[i].append(sum(rewards_temp[i]))
                epsilon_history.append(agents[0].epsilon)
                break

            if steps >= max_steps_done:
                if episode % 1000 == 0:
                    for i in range(n_agents):
                        episode_rewards[i].append(sum(rewards_temp[i]))
                        # print(sum(rewards_temp[i]))
                epsilon_history.append(agents[0].epsilon)
                break

            states = next_states

    return agents, episode_rewards, epsilon_history

def visualise_Qlearn_agents(agents, n_episodes=1):
    n_agents = len(agents)
    env = Food_Collector_Env(grid_size=[11, 11], n_agents=n_agents)

    # run n_episodes episodes
    frames = []
    for _ in tqdm(range(n_episodes)):
        # get initial state and actions
        states = env.reset()
        for i in range(n_agents):
            action = agents[i].agent_start(states[i])
        rewards = [0 for _ in range(n_agents)]
        step_counter = 0

        while True:
            step_counter += 1
            actions = []
            for i in range(n_agents):
                # action = agents[i].agent_step(rewards[i], states[i])
                action = agents[i].test_step(states[i])
                actions.append(action)
            next_states, rewards, done, info = env.step(actions)
            states = next_states

            frame = env.render()
            frames.append(frame)

            if done[0]:
                break
            if step_counter >= MAX_STEPS_DONE // 2:
                break

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

    agents, reward_history, epsilon_history = train_QL_agents(N_AGENTS, NUM_EPISODES, MAX_STEPS_DONE, EPS_DECAY,
                                                              EPS_MIN, STEP_SIZE, STEP_SIZE_VAR, GAMMA)
    team_rewards = [sum(x) for x in zip(*reward_history)]
    frames = visualise_Qlearn_agents(agents, n_episodes=3)
    frames = [resize_frame(frame, scale=16) for frame in frames]

    with open(f'./outputs/ctdiql_zeta{ZETA}_reward_{train_start_time}.pkl', 'wb') as f:
        pickle.dump(team_rewards, f)
    imageio.mimsave(f'./outputs/ctdiql_zeta{ZETA}_video_{train_start_time}.mp4', frames, fps=2)


