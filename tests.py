import pettingzoo
import random
import numpy as np
from pettingzoo.butterfly import cooperative_pong_v5
import imageio

def pettingzoo_env_test():
    env = cooperative_pong_v5.env(render_mode='human')
    env.reset(seed=42)
    frames = []

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        # print(type(observation), observation.shape)
        frames.append(observation)

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        # print(agent, action, reward)

        env.step(action)
    env.close()

    imageio.mimsave('./outputs/game_test.mp4', frames)

def pettingzoo_env_test2():
    from pettingzoo.atari import entombed_cooperative_v3

    env = entombed_cooperative_v3.env(render_mode="human")
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
