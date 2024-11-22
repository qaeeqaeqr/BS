import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import matplotlib
matplotlib.use('TkAgg')

class SiouxFallsNetwork(object):
    """
    对于Sioux Falls，知道其下面init中出现的信息即可。
    """
    def __init__(self, link_distrib_path, node_path):
        self.num_states = 0
        self.num_actions = 0
        # 每条链路上的*高斯分布*的均值和标准差
        self.mean_value: np.ndarray = None
        self.std_value: np.ndarray = None
        # 状态转移矩阵
        self.A: np.ndarray = None

        self.link_distrib_path = link_distrib_path
        self.node_path = node_path


    def build_network(self):
        """
        读取路网的节点、路径分布文件，获得需要的信息
        """
        node_file, link_file = open(self.node_path), open(self.link_distrib_path)
        node_reader, link_reader = csv.reader(node_file), csv.reader(link_file)

        # 统计节点数与行动数 （行动数即链路数）
        num_states = -1
        for row in node_reader:
            num_states += 1

        num_actions = -1
        for row in link_reader:
            num_actions += 1

        mean_value = np.zeros(shape=(num_actions, ))
        std_value = np.zeros(shape=(num_actions, ))
        A = np.zeros(shape=(num_states, num_actions))

        del link_reader
        link_file1 = open(self.link_distrib_path)
        link_reader = csv.reader(link_file1)
        next(link_reader)

        # 确定每个节点的分布以及状态转移矩阵
        for i, row in enumerate(link_reader):
            # print(row)
            mean_value[i] = float(row[2])
            std_value[i] = float(row[3])

            A[int(row[0])-1][i] = 1
            A[int(row[1])-1][i] = -1

        self.num_states = num_states
        self.num_actions = num_actions
        self.mean_value = mean_value
        self.std_value = std_value
        self.A = A

        # plt.matshow(self.A)
        # plt.show()

        node_file.close()
        link_file.close()

    def visualization(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(np.arange(1, self.num_states + 1))

        node_file, link_file = open(self.node_path), open(self.link_distrib_path)
        node_reader, link_reader = csv.reader(node_file), csv.reader(link_file)
        next(link_reader)
        for row in link_reader:
            G.add_edge(int(row[0]), int(row[1]), label=row[2] + ' ' + row[3])

        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos, with_labels=True,
                arrows=True, arrowstyle='-|>')
        edge_labels = nx.get_edge_attributes(G, 'label')
        # for (u, v, key), label in edge_labels.items():
        #     label_pos = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
        #     plt.text(label_pos[0], label_pos[1] + 1, label)
        plt.show()

        node_file.close()
        link_file.close()


if __name__ == '__main__':
    s = SiouxFallsNetwork(link_distrib_path='../Networks_data/MySiouxFalls_link_distrib.csv',
                          node_path='../Networks_data/MySiouxFalls_node.csv')
    s.build_network()
    s.visualization()
