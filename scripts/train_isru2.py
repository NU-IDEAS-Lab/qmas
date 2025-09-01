from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-hybridObs-e1h1p0-e200-r20rand-512x4",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env",
    "--user_name", "ideas-mas",

	"--num_extractors", "1",
    "--num_haulers", "1",
    "--num_prospectors", "0",

    "--hauler_capacity", "5",

    # "--lr", "3e-4",
    # "--critic_lr", "3e-4",
    # "--gae_lambda", "1.0",
    # "--gamma", "0.999",
    # "--entropy_coef", "0.05",
    # "--value_loss_coef", "0.5",

    "--world_size", "5",
    "--num_obstacles", "0",
    "--num_resources", "20",
    "--randomize_num_resources",

    "--observation_radius", "999999",
    # "--observation_mask",
    "--available_actions_mask",

    "--num_env_steps", "5000000",
    "--episode_length", "200",
    "--max_cycles", "200",

    # "--algorithm_class", "qmas.variant_two_modules.algorithm.QmasAlgorithm",
    # "--policy_class", "qmas.variant_two_modules.policy.QmasPolicy",
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

    "--seed", "0",

    "--n_rollout_threads", "300",
    "--cuda",
    "--cuda_idx", "3",

    "--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)