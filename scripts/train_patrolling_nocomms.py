
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
	"--experiment_name", "patrolling-obs200-gnn1",
	# "--experiment_name", "diffuser-patrolling-obs200-uqEnsemble3-pygNS-injectionAppend",
	"--project_name", "qmas",
	"--env_class", "patrolling_zoo.patrolling_zoo_v0.parallel_env",
	"--user_name", "ideas-mas",

	"--num_agents", "4",
	"--agent_speed", "40.0",
	"--action_method", "neighbors",
	"--observe_method", "pyg",
	"--observe_method_global", "adjacency",
	"--observation_radius", "200.0",
	"--communication_model", "none",
	"--communication_probability", "0.0",
	"--alpha", "1.0",
	"--beta", "0.5",
	"--reward_method_terminal", "average",

	"--graph_name", "milwaukee",
	"--graph_file", "../patrolling_zoo/graphs/milwaukee.graph",
	"--num_env_steps", "1000000",
	"--episode_length", "200",
	"--max_cycles", "200",
    "--num_mini_batch", "50",

	"--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
	"--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
	"--algorithm_name", "mappo",
	"--use_gnn_policy",
	"--use_gnn_mlp_policy",
	"--gnn_layer_N", "1",
	"--gnn_hidden_size", "128",
	"--gnn_skip_connections",
	"--gnn_neighbor_scoring",
	# "--use_recurrent_policy",
	"--use_centralized_V",
	"--use_gae",
	"--use_gae_amadm",
	"--skip_steps",
	"--share_policy",
	"--use_ReLU",
	"--hidden_size", "512",
	"--layer_N", "2",
	"--seed", "0",

	"--diffusion_model_type", "dit1d",
    "--prediction_uq_method", "ensemble",
    # "--prediction_uq_injection_method", "append",
    "--prediction_disable",
	"--prediction_ensemble_size", "3",
    "--state_encoder",

	"--n_rollout_threads", "99",
	"--cuda",
	"--cuda_idx", "7",
    "--cuda_idx_predictor", "7",
    "--threaded_training",
    "--n_training_threads", "16",

	"--save_interval", "10000",
	"--results_dir", "/data/group/mas/qmas/results",
	"--use_wandb",
]

main(args)
