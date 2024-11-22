from CTD import *
from argparser import args
from Networks.SiouxFalls import *

if __name__ == "__main__":
    num_states = 6  # Example with more than four nodes
    num_actions = 5  # Example with more than four actions

    mean_value = np.arange(1, num_actions + 1)
    std_value = (4 / 3) / np.arange(1, num_actions + 1) + (2 / 3)
    # Initialize matrices
    A = np.zeros((num_states, num_actions))
    start_node = np.arange(num_actions)
    end_node = start_node + 1

    for i in range(num_actions):
        A[start_node[i], i] = 1
        if end_node[i] < num_states:
            A[end_node[i], i] = -1
    print(A)
    origin = 0
    destination = num_states - 1

    ctd = CTDBase()
    performance = ctd.ctd(num_states, num_actions, mean_value, std_value, A, origin, destination, args)
    q_mean_var_table = np.load(args.save_path)
    q_mean_table, q_variance_table = q_mean_var_table[0], q_mean_var_table[1]

    print("\nFinal performance:\n", performance)
    print('-' * 30)
    print("\nFinal Q-Mean Table:\n", q_mean_table)
    print('-' * 30)
    print("\nFinal Q-Variance Table:\n", q_variance_table)
