from pettingzoo.sisl import pursuit_v4
from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.classic import connect_four_v3

from configs import alg_config, env_config

from algorithms.IQL import train_iql, IQL
from algorithms.CTD_IQL import train_ctdiql, CTDIQL
from algorithms.VDN import train_vdn, VDNAgent
from algorithms.CTD_VDN import train_ctdvdn, CTDVDNAgent

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


def train_IQL_on_pursuit():
    training_env = pursuit_v4.env(
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
    training_iql = IQL(num_agents=pursuit.n_pursuers,
                       state_dim=(pursuit.obs_range ** 2) * 3,
                       action_dim=pursuit.n_actions,
                       buffer_size=alg_config.buffer_size,
                       lr=alg_config.lr,
                       gamma=alg_config.gamma,
                       epsilon=alg_config.epsilon,
                       epsilon_decay=alg_config.epsilon_decay,
                       epsilon_min=alg_config.epsilon_min,
                       batch_size=alg_config.batch_size,
                       device=alg_config.device, )

    train_iql(training_env, training_iql,
              num_episodes=alg_config.num_episodes,
              seed=alg_config.train_seed,
              env_name='train IQL in pursuit',
              save_path=alg_config.IQL_persuit_model_path,
              load_path=None,
              fig_path=alg_config.output_dir)


def train_IQL_on_pong():
    training_env = cooperative_pong_v5.env(
        ball_speed=pong.ball_speed,
        left_paddle_speed=pong.left_paddle_speed,
        right_paddle_speed=pong.right_paddle_speed,
        cake_paddle=pong.cake_paddle,
        max_cycles=pong.max_cycles,
        bounce_randomness=pong.bounce_randomness,
        max_reward=pong.max_reward,
        off_screen_penalty=pong.off_screen_penalty,
        render_mode='human',  # pong环境有bug。其它的render_mode会导致observation啥也没有。
        render_fps=60,
    )

    training_iql = IQL(num_agents=2,
                       state_dim=15,  # 提取的特征
                       action_dim=pong.n_actions,
                       buffer_size=alg_config.buffer_size,
                       lr=alg_config.lr,
                       gamma=alg_config.gamma,
                       epsilon=alg_config.epsilon,
                       epsilon_decay=alg_config.epsilon_decay,
                       epsilon_min=alg_config.epsilon_min,
                       batch_size=alg_config.batch_size,
                       device=alg_config.device, )

    train_iql(training_env, training_iql,
              num_episodes=alg_config.num_episodes,
              seed=alg_config.train_seed,
              env_name='train IQL in Pong',
              save_path=alg_config.IQL_pong_model_path,
              load_path=None,
              fig_path=alg_config.output_dir)


def train_IQL_on_connect4():
    training_env = connect_four_v3.env(
        render_mode=connect4.render_mode
    )

    training_iql = IQL(num_agents=2,
                       state_dim=6 * 7 * 2,
                       action_dim=connect4.n_actions,
                       buffer_size=alg_config.buffer_size,
                       lr=alg_config.lr,
                       gamma=alg_config.gamma,
                       epsilon=alg_config.epsilon,
                       epsilon_decay=alg_config.epsilon_decay,
                       epsilon_min=alg_config.epsilon_min,
                       batch_size=alg_config.batch_size,
                       device=alg_config.device, )
    # note 暂不完成。因为agent与环境交互有额外限制，需要编写额外代码。


def train_CTDIQL_on_pursuit():
    training_env = pursuit_v4.env(
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
    training_ctdiql = CTDIQL(num_agents=pursuit.n_pursuers,
                             state_dim=(pursuit.obs_range ** 2) * 3,
                             action_dim=pursuit.n_actions,
                             buffer_size=alg_config.buffer_size,
                             lr=alg_config.lr,
                             gamma=alg_config.gamma,
                             epsilon=alg_config.epsilon,
                             epsilon_decay=alg_config.epsilon_decay,
                             epsilon_min=alg_config.epsilon_min,
                             batch_size=alg_config.batch_size,
                             device=alg_config.device,
                             zeta=alg_config.zeta,
                             lr_var=alg_config.lr_var, )

    train_ctdiql(training_env, training_ctdiql,
                 num_episodes=alg_config.num_episodes,
                 seed=alg_config.train_seed,
                 env_name='train CTD-IQL in pursuit',
                 save_path=alg_config.CTDIQL_persuit_model_path,
                 load_path=None,
                 fig_path=alg_config.output_dir)


def train_VDN_on_pursuit():
    training_env = pursuit_v4.env(
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
    training_vdn = VDNAgent(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=alg_config.epsilon,
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device, )

    train_vdn(training_env, training_vdn,
              num_episodes=alg_config.num_episodes,
              num_agents=pursuit.n_pursuers,
              seed=alg_config.train_seed,
              env_name='train VDN in pursuit',
              save_path=alg_config.VDN_persuit_model_path,
              load_path=None,
              fig_path=alg_config.output_dir)

def train_CTDVDN_on_pursuit():
    training_env = pursuit_v4.env(
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
    training_ctdvdn = CTDVDNAgent(num_agents=pursuit.n_pursuers,
                            state_dim=(pursuit.obs_range ** 2) * 3,
                            action_dim=pursuit.n_actions,
                            buffer_size=alg_config.buffer_size,
                            lr=alg_config.lr,
                            gamma=alg_config.gamma,
                            epsilon=alg_config.epsilon,
                            epsilon_decay=alg_config.epsilon_decay,
                            epsilon_min=alg_config.epsilon_min,
                            batch_size=alg_config.batch_size,
                            device=alg_config.device,
                            zeta=alg_config.zeta,
                            lr_var=alg_config.lr_var, )

    train_ctdvdn(training_env, training_ctdvdn,
              num_episodes=alg_config.num_episodes,
              num_agents=pursuit.n_pursuers,
              seed=alg_config.train_seed,
              env_name='train CTD_VDN in pursuit',
              save_path=alg_config.CTDVDN_persuit_model_path,
              load_path=None,
              fig_path=alg_config.output_dir)



if __name__ == "__main__":
    train_IQL_on_pong()
