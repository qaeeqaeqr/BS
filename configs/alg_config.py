import os
from torch import cuda
import pathlib

base_dir = pathlib.Path(__file__).parent.parent

# MARL
buffer_size = 10000
lr = 1e-3
gamma = 0.99
epsilon = 0.95
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'

# CTD
zeta = 0.1  # 风险系数
le_var = 1e-5

# train and test(visualize)
num_episodes = 15
train_seed = 42
test_seed = 42

# paths
output_dir = os.path.join(base_dir, 'outputs')
IQL_persuit_model_path = os.path.join(base_dir, 'models/IQL_persuit.pt')
IQL_pong_model_path = os.path.join(base_dir, 'models/IQL_pong.pt')
IQL_connect4_model_path = os.path.join(base_dir, 'models/IQL_connect4.pt')
CTDIQL_persuit_model_path = os.path.join(base_dir, 'models/CTDIQL_persuit.pt')
CTDIQL_pong_model_path = os.path.join(base_dir, 'models/CTDIQL_pong.pt')
CTDIQL_connect4_model_path = os.path.join(base_dir, 'models/CTDIQL_connect4.pt')
