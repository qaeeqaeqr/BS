from CTD import CTDBase

import numpy as np

class CTD4SiouxFalls(CTDBase):
    def __init__(self):
        super(CTD4SiouxFalls, self).__init__()

    def epsilon_greedy_action(self, last_state, state, q_table, A, destination, epsilon):
        # note 发现问题：程序给出的规划会在两个节点之间一直跳（如下所示）
        #  [1, 2, 6, 8, 7, 18, 7, 18, 7, 18, 7, 18, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 15]
        # note 这里需要考虑一个问题：程序也许需要保证不选择来时候的路，即不会在两个节点之间重复走。
        rand_number = np.random.rand()
        actions = np.where(A[state, :] == 1)[0]
        # print('last state:', last_state, 'current state:', state, end='\t')

        # 由于SiouxFalls的特性:每两个相邻节点之间都有来回的路，且每个节点至少和其它两个节点相邻。
        # 所以从所有可能的决策中删除回源节点的路。
        to_delete = None
        for a in actions:
            if np.where(A[:, a] == -1)[0] == last_state:  # 下一个节点是来时节点
                to_delete = a
                break
        actions = np.delete(actions, np.where(actions == to_delete))

        if state == destination:
            action = destination - 1
        else:
            if rand_number >= epsilon:
                action_q_values = q_table[state, actions]
                action = actions[np.argmin(action_q_values)]
            else:
                action = np.random.choice(actions)
        # print(action)
        return action
