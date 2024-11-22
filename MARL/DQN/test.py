import gymnasium as gym
from network import DQN
from args import *

import torch

env = gym.make("CartPole-v1", render_mode='human')

def test():
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions)
    policy_net.load_state_dict(torch.load(MODEL_PATH))

    done = False
    while not done:
        env.render()

        with torch.no_grad():
            action = policy_net(torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                                ).max(1).indices.view(1, 1)

        next_state, reward, done, truncated, info = env.step(action.item())
        state = next_state



if __name__ == '__main__':
    test()
