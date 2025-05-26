from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
	"--experiment_name", "simple-adversary",

    "--project_name", "qmas",
	"--env_class", "toy_problem.env.simpler_adversary.parallel_env",
	# "--env_class", "toy_problem.env.simple_adversary.parallel_env",
	# "--env_class", "toy_problem.toy_problem_v0.parallel_env",
	"--user_name", "ideas-mas",

	"--num_agents", "3",
    "--state_per_agent",
    
	"--num_env_steps", "1000000",
	"--episode_length", "200",
	# "--episode_allow_env_reset",
	"--max_cycles", "200",

	"--algorithm_class", "qmas.variant_two_modules.algorithm.QmasAlgorithm",
	"--policy_class", "qmas.variant_two_modules.policy.QmasPolicy",
	"--algorithm_name", "mappo",
	"--use_recurrent_policy",
    "--recurrent_N", "2",
    "--data_chunk_length", "3",
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
	"--cuda_idx", "2",

	"--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
	"--use_wandb"
]

main(args)