from pettingzoo.sisl import pursuit_v4
from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.classic import connect_four_v3

from configs import alg_config, env_config

from algorithms.IQL import visualize_iql, IQL
from algorithms.CTD_IQL import visualize_ctdiql, CTDIQL
from algorithms.VDN import visualize_vdn, VDNAgent
from algorithms.CTD_VDN import visualize_ctdvdn, CTDVDNAgent

pursuit = env_config.PersuitEnvConfig()
pong = env_config.PongEnvConfig()
connect4 = env_config.Connect4EnvConfig()

pursuit_env = pursuit_v4.env(
    max_cycles=pursuit.max_cycles,
    x_size=pursuit.x_size,
    y_size=pursuit.y_size,
    shared_reward=pursuit.shared_reward,
    n_evaders=pursuit.n_evaders,
    n_pursuers=pursuit.n_pursuers,
    obs_range=pursuit.obs_range,
    n_catch=pursuit.n_catch,
    freeze_evaders=pursuit.freeze_evaders,
    tag_reward=pursuit.tag_reward,
    catch_reward=pursuit.catch_reward,
    urgency_reward=pursuit.urgency_reward,
    surround=pursuit.surround,
    constraint_window=pursuit.constraint_window,
    render_mode=pursuit.render_mode,
)

pong_env = cooperative_pong_v5.env(
    ball_speed=pong.ball_speed,
    left_paddle_speed=pong.left_paddle_speed,
    right_paddle_speed=pong.right_paddle_speed,
    cake_paddle=pong.cake_paddle,
    max_cycles=pong.max_cycles,
    bounce_randomness=pong.bounce_randomness,
    max_reward=pong.max_reward,
    off_screen_penalty=pong.off_screen_penalty,
    render_mode=pong.render_mode,
)

connect4_env = connect_four_v3.env(
    render_mode=connect4.render_mode
)


def visualize_IQL_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_iql = IQL(num_agents=pursuit.n_pursuers,
                      state_dim=(pursuit.obs_range ** 2) * 3,
                      action_dim=pursuit.n_actions,
                      buffer_size=alg_config.buffer_size,
                      lr=alg_config.lr,
                      gamma=alg_config.gamma,
                      epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                      epsilon_decay=alg_config.epsilon_decay,
                      epsilon_min=alg_config.epsilon_min,
                      batch_size=alg_config.batch_size,
                      device=alg_config.device, )
    visualize_iql(testing_env, testing_iql,
                  seed=alg_config.test_seed,
                  env_name='visualize IQL in pursuit',
                  load_path=alg_config.IQL_persuit_model_path,
                  video_path=alg_config.output_dir)


def visualize_IQL_on_pong():
    testing_env = cooperative_pong_v5.env(
        ball_speed=pong.ball_speed,
        left_paddle_speed=pong.left_paddle_speed,
        right_paddle_speed=pong.right_paddle_speed,
        cake_paddle=pong.cake_paddle,
        max_cycles=pong.max_cycles,
        bounce_randomness=pong.bounce_randomness,
        max_reward=pong.max_reward,
        off_screen_penalty=pong.off_screen_penalty,
        render_mode=pong.render_mode,
    )

    testing_iql = IQL(num_agents=2,
                      state_dim=280 * 480 * 3,
                      action_dim=pong.n_actions,
                      buffer_size=alg_config.buffer_size,
                      lr=alg_config.lr,
                      gamma=alg_config.gamma,
                      epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                      epsilon_decay=alg_config.epsilon_decay,
                      epsilon_min=alg_config.epsilon_min,
                      batch_size=alg_config.batch_size,
                      device=alg_config.device, )

    visualize_iql(testing_env, testing_iql,
                  seed=alg_config.test_seed,
                  env_name='visualize IQL in pong',
                  load_path=alg_config.IQL_pong_model_path,
                  video_path=alg_config.output_dir)


def visualize_CTDIQL_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_ctdiql = CTDIQL(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,
                            zeta=alg_config.zeta,
                            lr_var=alg_config.lr_var, )
    visualize_ctdiql(testing_env, testing_ctdiql,
                     seed=alg_config.test_seed,
                     env_name='visualize IQL in pursuit',
                     load_path=alg_config.IQL_persuit_model_path,
                     video_path=alg_config.output_dir)

def visualize_VDN_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_vdn = VDNAgent(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,)

    visualize_vdn(testing_env, testing_vdn,
                  seed=alg_config.test_seed,
                  env_name='visualize VDN in pursuit',
                  load_path=alg_config.VDN_persuit_model_path,
                  video_path=alg_config.output_dir)

def visualize_CTDVDN_on_pursuit():
    testing_env = pursuit_v4.env(
        max_cycles=pursuit.max_cycles,
        x_size=pursuit.x_size,
        y_size=pursuit.y_size,
        shared_reward=pursuit.shared_reward,
        n_evaders=pursuit.n_evaders,
        n_pursuers=pursuit.n_pursuers,
        obs_range=pursuit.obs_range,
        n_catch=pursuit.n_catch,
        freeze_evaders=pursuit.freeze_evaders,
        tag_reward=pursuit.tag_reward,
        catch_reward=pursuit.catch_reward,
        urgency_reward=pursuit.urgency_reward,
        surround=pursuit.surround,
        constraint_window=pursuit.constraint_window,
        render_mode='rgb_array',
    )
    testing_ctdvdn = CTDVDNAgent(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=0,  # 测试时，算法应该完全按照学习的策略选择动作。
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,
                            zeta=alg_config.zeta,
                            lr_var=alg_config.lr_var, )

    visualize_ctdvdn(testing_env, testing_ctdvdn,
                     seed=alg_config.test_seed,
                     env_name='visualize CTD_VDN in pursuit',
                     load_path=alg_config.CTDVDN_persuit_model_path,
                     video_path=alg_config.output_dir)

if __name__ == '__main__':
    visualize_CTDVDN_on_pursuit()
