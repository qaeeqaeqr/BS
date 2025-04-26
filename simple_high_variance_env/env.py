import numpy as np
from PIL import Image, ImageDraw
from gym import spaces


class FoodCollectEnv:
    def __init__(self):
        self.grid_size = 11
        self.n_agents = 2
        self.state = [0, 0]  # 初始化第一个智能体位置
        self.state2 = [0, 0]  # 初始化第二个智能体位置
        self.treasure = [self.grid_size-1, self.grid_size-1]  # 宝藏位置

    def reset(self):
        self.state = [9, 8]
        self.state2 = [10, 10]
        return self._get_obs()

    def step(self, actions):
        # 处理两个智能体的动作
        action1, action2 = actions[0], actions[1]
        # 智能体1移动
        if action1 == 0:  # 上
            self.state[1] = max(0, self.state[1] - 1)
        elif action1 == 1:  # 下
            self.state[1] = min(self.grid_size - 1, self.state[1] + 1)
        elif action1 == 2:  # 左
            self.state[0] = max(0, self.state[0] - 1)
        elif action1 == 3:  # 右
            self.state[0] = min(self.grid_size - 1, self.state[0] + 1)

        # 智能体2移动
        if action2 == 0:  # 上
            self.state2[1] = max(0, self.state2[1] - 1)
        elif action2 == 1:  # 下
            self.state2[1] = min(self.grid_size - 1, self.state2[1] + 1)
        elif action2 == 2:  # 左
            self.state2[0] = max(0, self.state2[0] - 1)
        elif action2 == 3:  # 右
            self.state2[0] = min(self.grid_size - 1, self.state2[0] + 1)

        done = [False, False]
        rewards = [-0.001, -0.001]

        # 检查智能体1是否找到宝藏
        if self.state == self.treasure or self.state2 == self.treasure:
            done[:] = [True, True]
            rewards[:] = [1, 1]

        return self._get_obs(), rewards, done, {}

    def render(self):
        # 返回环境的图像表示
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[tuple(self.treasure)] = 1  # 宝藏位置标记为1
        grid[tuple(self.state)] += 0.5  # 智能体1位置标记为0.5
        grid[tuple(self.state2)] += 0.5  # 智能体2位置标记为0.5
        return grid

    def _get_obs(self):
        # 获取每个智能体的观测
        obs = []
        # 智能体1的观测
        agent1_pos = self.state
        treasure_pos = self.treasure
        relative_pos1 = np.concatenate([np.array(agent1_pos), np.array(treasure_pos)])
        obs.append(relative_pos1)
        # 智能体2的观测
        agent2_pos = self.state2
        treasure_pos = self.treasure
        relative_pos2 = np.concatenate([np.array(agent2_pos), np.array(treasure_pos)])
        obs.append(relative_pos2)
        return obs

    @property
    def observation_space(self):
        # 每个智能体的观测空间
        return [spaces.Box(low=0, high=self.grid_size, shape=(4,)) for _ in range(self.n_agents)]

    @property
    def action_space(self):
        # 每个智能体的动作空间
        return [spaces.Discrete(4) for _ in range(self.n_agents)]


def visualize_env(env, grid_size=10, cell_size=50):
    """
    可视化环境，返回一张图像。

    参数:
        env: 环境对象，包含智能体和宝藏的位置信息
        grid_size: 网格大小，默认为10x10
        cell_size: 每个格子的像素大小，默认为50

    返回:
        image: PIL图像对象
    """
    # 创建一个黑色背景的图像
    image = Image.new("RGB", ((grid_size + 2) * cell_size, (grid_size + 2) * cell_size), "black")
    draw = ImageDraw.Draw(image)

    # 绘制墙壁（灰色）
    wall_color = "grey"
    for i in range(grid_size + 2):
        # 上下墙壁
        draw.rectangle(
            [
                (i * cell_size, 0),
                ((i + 1) * cell_size, cell_size)
            ],
            fill=wall_color
        )
        draw.rectangle(
            [
                (i * cell_size, (grid_size + 1) * cell_size),
                ((i + 1) * cell_size, (grid_size + 2) * cell_size)
            ],
            fill=wall_color
        )
    for i in range(grid_size + 2):
        # 左右墙壁
        draw.rectangle(
            [
                (0, i * cell_size),
                (cell_size, (i + 1) * cell_size)
            ],
            fill=wall_color
        )
        draw.rectangle(
            [
                ((grid_size + 1) * cell_size, i * cell_size),
                ((grid_size + 2) * cell_size, (i + 1) * cell_size)
            ],
            fill=wall_color
        )

    # 绘制宝藏（绿色）
    treasure_pos = tuple(env.treasure)
    draw.rectangle(
        [
            ((treasure_pos[0] + 1) * cell_size, (treasure_pos[1] + 1) * cell_size),
            ((treasure_pos[0] + 2) * cell_size, (treasure_pos[1] + 2) * cell_size)
        ],
        fill="green"
    )
    # 绘制智能体1（红色）
    agent1_pos = tuple(env.state)
    draw.rectangle(
        [
            ((agent1_pos[0] + 1) * cell_size, (agent1_pos[1] + 1) * cell_size),
            ((agent1_pos[0] + 2) * cell_size, (agent1_pos[1] + 2) * cell_size)
        ],
        fill="red"
    )

    # 绘制智能体2（橙色）
    agent2_pos = tuple(env.state2)
    draw.rectangle(
        [
            ((agent2_pos[0] + 1) * cell_size, (agent2_pos[1] + 1) * cell_size),
            ((agent2_pos[0] + 2) * cell_size, (agent2_pos[1] + 2) * cell_size)
        ],
        fill="orange"
    )


    return image

if __name__ == "__main__":
    env = FoodCollectEnv()
    env.reset()
    img = visualize_env(env, grid_size=env.grid_size, cell_size=30)
    img.save('./outputs/env.png')
