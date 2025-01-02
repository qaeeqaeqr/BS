class PersuitEnvConfig:
    max_cycles = 500
    x_size = 16
    y_size = 16
    shared_reward = True
    n_evaders = 30
    n_pursuers = 8
    obs_range = 7
    n_catch = 2
    freeze_evaders = False
    tag_reward = 0.01
    catch_reward = 5.0
    urgency_reward = -0.1
    surround = True
    constraint_window = 1.0
    render_mode = 'rgb_array'

    n_actions = 5  # 上下左右或静止不动


class PongEnvConfig:
    ball_speed = 9
    left_paddle_speed = 12
    right_paddle_speed = 12
    cake_paddle = True
    max_cycles = 900
    bounce_randomness = False
    max_reward = 100
    off_screen_penalty = -10
    render_mode = 'rgb_array'

    n_actions = 3  # 往上或往下或静止


class Connect4EnvConfig:
    render_mode = None

    # 这个环境智能体只输出一个代表将要下棋的位置的序号的数字就可以了，但为了更好利用神经网络，故让智能体对每个位置都输出值。
    n_actions = 6 * 7

