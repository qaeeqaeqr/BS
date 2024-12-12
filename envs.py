from pettingzoo.sisl import pursuit_v4
from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.classic import connect_four_v3

def persuit_env(
    max_cycles=500,
    x_size=16,
    y_size=16,
    shared_reward=True,
    n_evaders=30,
    n_pursuers=8,
    obs_range=7,
    n_catch=2,
    freeze_evaders=False,
    tag_reward=0.01,
    catch_reward=5.0,
    urgency_reward=-0.1,
    surround=True,
    constraint_window=1.0,
    render_mode=None,
):
    return pursuit_v4.env(
        max_cycles=max_cycles,
        x_size=x_size,
        y_size=y_size,
        shared_reward=shared_reward,
        n_evaders=n_evaders,
        n_pursuers=n_pursuers,
        obs_range=obs_range,
        n_catch=n_catch,
        freeze_evaders=freeze_evaders,
        tag_reward=tag_reward,
        catch_reward=catch_reward,
        urgency_reward=urgency_reward,
        surround=surround,
        constraint_window=constraint_window,
        render_mode=render_mode,
    )


def pong_env(
    ball_speed=9,
    left_paddle_speed=12,
    right_paddle_speed=12,
    cake_paddle=True,
    max_cycles=900,
    bounce_randomness=False,
    max_reward=100,
    off_screen_penalty=-10,
    render_mode=None,
):
    return cooperative_pong_v5.env(
        ball_speed=ball_speed,
        left_paddle_speed=left_paddle_speed,
        right_paddle_speed=right_paddle_speed,
        cake_paddle=cake_paddle,
        max_cycles=max_cycles,
        bounce_randomness=bounce_randomness,
        max_reward=max_reward,
        off_screen_penalty=off_screen_penalty,
        render_mode=render_mode,
    )

def connect4_env(  # 这个环境的参数是写死的，改不了。
    render_mode=None,
):
    return connect_four_v3.env(
        render_mode=render_mode,
    )