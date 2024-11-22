from argparser import args
from CTD4Networks import CTD4SiouxFalls

from Networks.SiouxFalls import SiouxFallsNetwork

import numpy as np
import matplotlib.pyplot as plt


def test_SiouxFalls(origin=1, destination=15, q_table_path='./Q-Tables/SiouxFalls.npy'):
    """
    :param origin: 给定的起点
    :param destination: 给定的终点
    :return: 找到的最优路径的mean和std
             用Q-Table判断找到的路径
    """
    s = SiouxFallsNetwork(link_distrib_path='./Networks_data/MySiouxFalls_link_distrib.csv',
                          node_path='./Networks_data/MySiouxFalls_node.csv')
    s.build_network()

    ctd_4_sioux_falls = CTD4SiouxFalls()

    origin, destination = origin - 1, destination - 1  # 编号问题。数组从0开始编号。

    print('Training...')
    final_performance = ctd_4_sioux_falls.ctd(
                              s.num_states,
                              s.num_actions,
                              s.mean_value,
                              s.std_value,
                              s.A,
                              origin,
                              destination,
                              args,
                              save_path=q_table_path,
    )
    print('\nFinish training!\n')

    q_table_mean_var = np.load(q_table_path)
    q_mean, q_var = q_table_mean_var[0], q_table_mean_var[1]
    q_table = q_mean + args.zeta * np.sqrt(q_var)
    # plt.matshow(q_table)
    # plt.show()

    # 下面根据训练好的Q-Table查看最终选择的最优路径
    state = origin
    action = ctd_4_sioux_falls.epsilon_greedy_action(state, state, q_table, s.A, destination, args.epsilon)
    path = [state + 1]

    while state != destination:
        next_state = ctd_4_sioux_falls.get_next_state(state, action, s.A, destination)
        next_action = ctd_4_sioux_falls.epsilon_greedy_action(state, next_state,
                                                              q_table, s.A, destination, args.epsilon)

        state = next_state
        action = next_action
        path.append(int(state) + 1)

    print('optimal path:', path)


if __name__ == '__main__':
    test_SiouxFalls()
