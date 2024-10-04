from onpolicy.scripts.train.train_flocking import get_config, parse_args, main

import os
import numpy as np
os.environ["WANDB__SERVICE_WAIT"] = "300"
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

parser = get_config()
all_args = parse_args([], parser)

all_args.experiment_name = "nosheep-matrix-random-graph-tanh"
all_args.env_name = "Flocking"
all_args.user_name = "ideas-mas"

all_args.randomize_graph = True
all_args.regenerate_graph_on_reset = True

all_args.num_shepherds = 1
all_args.num_sheep = 0

all_args.sheep_force_avoid_shepherd = 10.0
all_args.sheep_force_avoid_sheep = 10.0
all_args.sheep_force_align = 0.0
all_args.sheep_force_cluster = 1.0

all_args.sheep_radius_avoid_shepherd = 20.0
all_args.sheep_radius_avoid_sheep = 5.0

all_args.speed_sheep = 1.0
all_args.speed_shepherds = 5.0
all_args.observe_method = "matrix"
all_args.observation_radius = np.inf
all_args.communication_model = "bernoulli"
all_args.communication_probability = 0.1
all_args.alpha = 1.0
all_args.beta = 10.0
all_args.reward_method_terminal = "none"

all_args.num_env_steps = 10e5 * 1 #total number of steps
all_args.episode_length = 1000 #number of steps in a training episode
all_args.max_cycles = all_args.episode_length #number of steps in an environment episode

all_args.algorithm_name = "rmappo"
all_args.use_gnn_policy = False
all_args.use_gnn_mlp_policy = True
all_args.use_gnn_critic = False
all_args.use_gnn_mlp_critic = True
all_args.gnn_layer_N = 1
all_args.gnn_hidden_size = 1024
all_args.gnn_skip_connections = False
all_args.use_recurrent_policy = True
all_args.use_naive_recurrent_policy = False
all_args.use_centralized_V = True
all_args.use_gae = True
all_args.use_gae_amadm = False
all_args.share_policy = True
all_args.sep_share_policy = False
all_args.share_reward = False
all_args.skip_steps_sync = False
all_args.skip_steps_async = False
all_args.use_ReLU = True
# all_args.lr = 1e-3
# all_args.entropy_coef = 0.1
all_args.hidden_size = 2048

all_args.n_rollout_threads = 10
all_args.save_interval = 100000
all_args.cuda = True
all_args.cuda_idx = 5

all_args.use_wandb = True



main([], parsed_args = all_args)