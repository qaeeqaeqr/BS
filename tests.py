import pettingzoo
import random
import numpy as np
from envs import persuit_env, pong_env, connect4_env

def pettingzoo_env_test():
    env = persuit_env()
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
    env.close()


if __name__ == "__main__":
    pettingzoo_env_test()
