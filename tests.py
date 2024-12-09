import pettingzoo
from pettingzoo.classic import tictactoe_v3
import random
import numpy as np

def pettingzoo_env_test():
    # 创建井字棋环境
    env = tictactoe_v3.env(render_mode="human")
    # 重置环境以开始新的游戏
    env.reset(seed=42)
    # 循环直到游戏结束
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # 随机选择一个可用的动作
            action = np.random.choice(np.where(mask)[0])

        env.step(action)

    # 渲染最终的环境状态
    env.render()

if __name__ == "__main__":
    pass
