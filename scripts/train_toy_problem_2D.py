from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
	"--experiment_name", "toy-problem-2D",
    "--project_name", "qmas",
	"--env_class", "toy_problem.toy_problem_2D_v0.parallel_env",
	"--user_name", "ideas-mas",

	"--num_agents", "1",
    "--state_per_agent",
	"--polynomial_degree", "0",

	"--num_env_steps", "1000000",
	"--episode_length", "10",
	"--max_cycles", "10",

	"--algorithm_class", "qmas.variant_two_modules.algorithm.QmasAlgorithm",
	"--policy_class", "qmas.variant_two_modules.policy.QmasPolicy",
	"--algorithm_name", "mappo",
	# "--use_recurrent_policy",
    # "--recurrent_N", "1",
    # "--data_chunk_length", "2",
	"--use_centralized_V",
	"--use_gae",
	"--use_gae_amadm",
    "--skip_steps",
	"--share_policy",
	"--no-share_reward",
	"--use_ReLU",
	"--hidden_size", "512",
	"--layer_N", "4",
	"--seed", "0",

	"--n_rollout_threads", "100",
	"--cuda",
	"--cuda_idx", "4",

	"--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
	"--use_wandb",
]

main(args)