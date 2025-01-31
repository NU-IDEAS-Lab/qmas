from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
	"--experiment_name", "toy-problem",
    "--project_name", "qmas",
	"--env_class", "toy_problem.toy_problem_v0.parallel_env",
	"--user_name", "ideas-mas",

	"--num_agents", "1",
    "--state_per_agent",

	"--num_env_steps", "1000000",
	"--episode_length", "50",
	"--max_cycles", "50",

	"--algorithm_class", "qmas.algorithm.QmasAlgorithm",
	"--policy_class", "qmas.variant_two_modules.policy.QmasPolicy",
	"--algorithm_name", "rmappo",
	"--use_recurrent_policy",
    "--recurrent_N", "1",
    "--data_chunk_length", "2",
	"--use_centralized_V",
	"--use_gae",
	"--use_gae_amadm",
    "--skip_steps",
	"--share_policy",
	"--no-share_reward",
	"--use_ReLU",
	"--hidden_size", "128",
	"--layer_N", "3",
	"--seed", "0",

	"--n_rollout_threads", "10",
	"--save_interval", "1000",
	"--cuda",
	"--cuda_idx", "0",

	"--use_wandb",
]

main(args)