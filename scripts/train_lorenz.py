from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "lorenz-obsprob0.5-normalizedObs-newParams-d6-tscale0.1",
    "--project_name", "qmas",
    "--env_class", "toy_problem.toy_problem_v1.parallel_env",
    "--user_name", "ideas-mas",

    "--num_agents", "1",
    "--num_adversaries", "1",
    "--num_dimensions", "3",
    "--observation_probability", "0.5",
    # "--observation_mask",
    "--world_size", "5",
    "--random_start_positions",
    "--adversary_dynamics", "lorenz",

    "--num_env_steps", "10000000",
    "--episode_length", "600",
    "--max_cycles", "600",
    "--num_mini_batch", "12",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_centralized_V",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "128",
    "--layer_N", "2",

    "--diffusion_model_type", "dit1d",
    "--prediction_ensemble_size", "1",

    "--seed", "0",

    "--n_rollout_threads", "400",
    "--cuda",
    "--cuda_idx", "3",

    "--save_interval", "100000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)