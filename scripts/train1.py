from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
	"--experiment_name", "pettingzoo-test",
    "--project_name", "qmas",
	"--env_name", "patrolling_zoo.patrolling_zoo_v0",
	"--user_name", "ideas-mas",

	"--num_agents", "4",
	"--agent_speed", "40.0",
	"--action_method", "neighbors",
	"--observe_method", "pyg",
	"--observe_method_global", "adjacency",
	"--observation_radius", "400.0",
	"--observation_bitmap_size", "40",
    "--state_per_agent",
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

	"--algorithm_name", "mappo",
	"--use_gnn_policy",
	"--use_gnn_mlp_policy",
	"--gnn_layer_N", "10",
	"--gnn_hidden_size", "128",
	"--gnn_skip_connections",
	"--use_recurrent_policy",
	"--no-use_naive_recurrent_policy",
	"--use_centralized_V",
	"--use_gae",
	"--use_gae_amadm",
	"--share_policy",
	"--no-share_reward",
	"--skip_steps",
	"--use_ReLU",
	"--hidden_size", "512",

	"--n_rollout_threads", "5",
	"--save_interval", "1000",
	"--cuda",
	"--cuda_idx", "2",

	"--use_wandb",
]

main(args)