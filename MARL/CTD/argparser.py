import argparse

def CTD_Argparser():
    parser = argparse.ArgumentParser(description="Arguments for CTD")

    parser.add_argument('--episodes',         type=int,   metavar='', default=50000,
                        help="CTD training episodes")
    parser.add_argument('--epsilon',          type=float, metavar='', default=0.5,
                        help="Epsilon in epsilon-greedy")
    parser.add_argument('--gamma',            type=int,   metavar='', default=1,
                        help="FIXED when applying RL to RSP problems.")
    parser.add_argument('--alpha_mean',       type=float, metavar='', default=1e-1,
                        help="Learning rate for Q-mean-table")
    parser.add_argument('--alpha_variance',   type=float, metavar='', default=1e-3,
                        help="Learning rate for Q-variance-table")
    parser.add_argument('--zeta',             type=float, metavar='', default=1,
                        help="Risk-averse parameter")
    parser.add_argument('--save_path',        type=str,   metavar='', default='./Q-mean-var-table.npy',
                        help="Path to saved Q-Table")

    args = parser.parse_args()
    return args

args = CTD_Argparser()
