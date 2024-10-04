import os
import time
from multiprocessing import Process

import numpy as np
import torch

from onpolicy.scripts.train.train_flocking import get_config, main, parse_args

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_flush_denormal(True)

os.environ["WANDB__SERVICE_WAIT"] = "300"

parser = get_config()
all_args = parse_args([], parser)

all_args.experiment_name = "random-matrix-single-no-trunc-nearby-reward-sweep"
all_args.env_name = "Flocking"
all_args.user_name = "ideas-mas"

all_args.randomize_graph = True
all_args.regenerate_graph_on_reset = True
all_args.random_graph_danger_zones = 0
all_args.random_graph_grid_size = 2
all_args.random_graph_grid_border = 10

all_args.num_shepherds = 2
all_args.num_sheep = 2

all_args.sheep_force_avoid_shepherd = 10.0
all_args.sheep_force_avoid_sheep = 10.0
all_args.sheep_force_align = 0.0
all_args.sheep_force_cluster = 1.0

all_args.sheep_radius_avoid_shepherd = 30.0
all_args.sheep_radius_avoid_sheep = 5.0

all_args.speed_sheep = 1.0
all_args.speed_shepherds = 5.0
all_args.observe_method = "graph"
all_args.observation_radius = np.inf
all_args.communication_model = "bernoulli"
all_args.communication_probability = 0.1
all_args.alpha = 0.5
all_args.beta = 0.01
all_args.reward_method_terminal = "none"

all_args.num_env_steps = 1e6 * 50  # total number of steps
# all_args.num_env_steps = 1e4 * 2  # total number of steps
# all_args.num_env_steps = 1e3 * 2  # total number of steps
all_args.episode_length = 100  # number of steps in a training episode
all_args.max_cycles = (
    all_args.episode_length
)  # number of steps in an environment episode

all_args.algorithm_name = "mappo"
all_args.use_gnn_policy = True
all_args.use_gnn_mlp_policy = True
all_args.use_gnn_critic = True
all_args.use_gnn_mlp_critic = True
all_args.gnn_layer_N = 1
all_args.gnn_hidden_size = 256
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
all_args.concat_k = 0 # Disable by setting to 0

# all_args.lr = 1e-6
# all_args.critic_lr = 1e-6
# all_args.entropy_coef = 1e-4

# all_args.lr = 2.5e-6
# all_args.critic_lr = 2.5e-6
# all_args.entropy_coef = 1e-4

# all_args.lr = 2.5e-6
# all_args.critic_lr = 2.5e-6
# all_args.entropy_coef = 1e-4
# all_args.use_linear_lr_decay = True

# all_args.lr = 1e-6
# all_args.critic_lr = 2.5e-6
# all_args.entropy_coef = 1e-4
# all_args.use_linear_lr_decay = True

all_args.lr = 5e-6
all_args.critic_lr = 5e-6
all_args.entropy_coef = 1e-4
all_args.use_linear_lr_decay = True

all_args.layer_N = 4
all_args.hidden_size = 128
all_args.use_orthogonal = False

# all_args.n_rollout_threads = 100
all_args.n_rollout_threads = 25
# all_args.n_rollout_threads = 2
all_args.save_interval = 100000
all_args.cuda = True
all_args.cuda_idx = 7

all_args.use_wandb = False

first_parameter = [4, 8]
second_parameter = [128, 512]

# first_parameter = [1e-5, 5e-6, 2.5e-6]
# second_parameter = [1e-5, 5e-6, 2.5e-6]
# second_parameter = [5e-6]

cuda_to_use = [2, 3]
# cuda_to_use = [1, 2, 3]

p_queue = []

# main([], parsed_args = all_args)
# quit()

for i, p1 in enumerate(first_parameter):
    for p2 in second_parameter:
        # all_args.lr = p2
        # all_args.critic_lr = p2

        all_args.layer_N = p1
        all_args.hidden_size = p2

        all_args.cuda_idx = 2
        all_args.experiment_name = (
            f"exp-11.010-graph-double-herding-{p1}-layers-{p2}-units"
        )

        # Multiprocessing
        p = Process(target=main, args=([],), kwargs={"parsed_args": all_args})
        p.start()
        p_queue.append(p)
        time.sleep(5)
        
        # # Profiling
        # import cProfile
        # cProfile.run('main([], parsed_args = all_args)', 'cprof')
        
        # # Normal
        # main([], parsed_args = all_args)

for p in p_queue:
    p.join()
