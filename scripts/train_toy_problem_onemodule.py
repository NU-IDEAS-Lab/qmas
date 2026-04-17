from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "toy-problem-onemodule-predObs",
    "--project_name", "qmas",
    "--env_class", "toy_problem.toy_problem_v1.parallel_env",
    "--user_name", "ideas-mas",

    "--num_agents", "1",
    "--num_adversaries", "1",
    "--num_dimensions", "2",
    "--observation_probability", "0.5",
    # "--observation_mask",
    "--random_start_positions",

    "--num_env_steps", "3000000",
    "--episode_length", "50",
    "--max_cycles", "50",

	"--algorithm_class", "qmas.variant_one_module.algorithm.QmasAlgorithmOneModule",
	"--policy_class", "qmas.variant_one_module.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_centralized_V",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "128",
    "--layer_N", "2",

    "--diffusion_model_type", "dit1d",
    "--prediction_ensemble_size", "3",
    # "--prediction_disable",
    # "--prediction_uq_method", "estimation",
    "--prediction_during_training",

    "--seed", "0",

    "--n_rollout_threads", "99",
    "--cuda",
    "--cuda_idx", "1",
    "--cuda_idx_predictor", "4",
    "--threaded_training",
    "--n_training_threads", "16",

    "--save_interval", "100000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)