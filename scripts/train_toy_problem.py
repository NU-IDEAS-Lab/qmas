from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "1module-toy-problem-ensemble3-obsprob0.5",
    "--project_name", "qmas",
    "--env_class", "toy_problem.toy_problem_v1.parallel_env",
    "--user_name", "ideas-mas",

    "--num_agents", "1",
    "--num_adversaries", "1",
    "--state_per_agent",
    "--num_dimensions", "2",
    "--observation_probability", "0.5",
    "--random_start_positions",

    "--num_env_steps", "3000000",
    "--episode_length", "50",
    "--max_cycles", "50",

    "--algorithm_class", "qmas.variant_one_module.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_one_module.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_centralized_V",
    "--use_gae",
    "--use_gae_amadm",
    "--skip_steps",
    "--share_policy",
    "--no-share_reward",
    "--use_ReLU",
    "--hidden_size", "512",
    "--layer_N", "4",

    "--diffusion_model_type", "dit1d",
    "--prediction_ensemble_size", "3",

    "--seed", "0",

    "--n_rollout_threads", "300",
    "--cuda",
    "--cuda_idx", "4",

    "--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)