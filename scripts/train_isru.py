from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env",
    "--user_name", "ideas-mas",

	"--num_extractors", "1",
    "--num_haulers", "1",
    "--num_prospectors", "1",

    "--hauler_capacity", "3",

    "--world_size", "10",
    "--observation_radius", "4",
    "--num_obstacles", "0",

    "--num_env_steps", "10000000",
    "--episode_length", "200",
    "--max_cycles", "200",

    "--algorithm_class", "qmas.variant_two_modules.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.policy.QmasPolicy",
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

    "--n_rollout_threads", "100",
    "--cuda",
    "--cuda_idx", "0",

    "--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)