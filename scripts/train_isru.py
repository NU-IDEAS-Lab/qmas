from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-UN-e1h1p0-e200-r20curr-h5-5x5-512x4",
    "--project_name", "qmas",
    # "--env_class", "isru_zoo.isru_v0.parallel_env",
    "--env_class", "isru_zoo.isru_v0.parallel_env_map_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "1",
    "--num_haulers", "1",
    "--num_prospectors", "0",

    "--hauler_capacity", "5",

    "--world_size", "5",
    "--num_obstacles", "0",
    "--num_resources", "20",
    # "--randomize_num_resources",
    "--curriculum_num_resources",

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
    "--no-use_ReLU",
    "--no-use_feature_normalization",
    # "--no-use_valuenorm",
    "--hidden_size", "128",
    "--layer_N", "2",

    "--diffusion_model_type", "dit1d",

    "--seed", "0",

    "--n_rollout_threads", "25",
    "--cuda",
    "--cuda_idx", "3",

    "--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)