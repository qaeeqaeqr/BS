import numpy as np
from scipy.stats import norm

class CTDBase(object):
    def __init__(self):
        pass

    def epsilon_greedy_action(self, last_state, state, q_table, A, destination, epsilon):
        rand_number = np.random.rand()
        actions = np.where(A[state, :] == 1)[0]

        if state == destination:
            action = destination - 1
        else:
            if rand_number >= epsilon:
                action_q_values = q_table[state, actions]
                action = actions[np.argmin(action_q_values)]
            else:
                action = np.random.choice(actions)
        return action

    def get_next_state(self, state, action, A, destination):
        return state if state == destination else np.where(A[:, action] == -1)[0][0]

    def get_reward(self, state, action, mean_value, std_value, destination):
        """
        reward即该条道路的实际通过时间（通过已知的分布采样得出）
        """
        return norm.rvs(mean_value[action], std_value[action])

    def ctd(self, num_states, num_actions, mean_value, std_value, A, origin, destination,
            args, save_path=''):
        episodes = args.episodes
        epsilon = args.epsilon
        gamma = args.gamma
        alpha_mean = args.alpha_mean
        alpha_variance = args.alpha_variance
        zeta = args.zeta

        covSigma = np.diag(std_value ** 2)

        q_mean_table = np.zeros((num_states, num_actions))
        q_variance_table = np.zeros((num_states, num_actions))
        q_table = q_mean_table + zeta * np.sqrt(q_variance_table)

        performances = [None]

        for episode_count in range(1, episodes + 1):
            state = origin
            action = self.epsilon_greedy_action(state, state, q_table, A, destination, epsilon)

            while state != destination:
                next_state = self.get_next_state(state, action, A, destination)
                next_action = self.epsilon_greedy_action(state, next_state, q_table, A, destination, epsilon)

                reward = self.get_reward(state, action, mean_value, std_value, destination)
                delta = reward + gamma * q_mean_table[next_state, next_action] - q_mean_table[state, action]
                q_mean_table[state, action] += alpha_mean * delta

                delta_variance = delta ** 2 + gamma * q_variance_table[next_state, next_action] - q_variance_table[state, action]
                q_variance_table[state, action] += alpha_variance * delta_variance

                q_table[state, action] = q_mean_table[state, action] + zeta * np.sqrt(q_variance_table[state, action])
                state = next_state
                action = next_action

                if state == destination:
                    _, path = min((q_table[0, a], a) for a in range(num_actions))
                    path_name = np.zeros(num_actions)
                    path_name[path] = 1
                    performance = mean_value @ path_name + zeta * np.sqrt(path_name @ covSigma @ path_name)
                    performances.append(float(performance))

            print('\r', episode_count, '/', episodes, 'episodes trained.', end='')

        if save_path:
            # 将q-mean-table和q-variance-table沿新的axis拼接起来并保存（这样只保存一个npy文件就好了）
            saving_q_table = np.stack((q_mean_table, q_variance_table), axis=0)
            np.save(save_path, saving_q_table)



