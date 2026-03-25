from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-UNGN-1superBot-e400-r20-h20-5x5-alwaysCommunicate",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_map_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "1",
    "--num_haulers", "0",
    "--num_prospectors", "0",

    "--hauler_capacity", "20",

    "--world_size", "5",
    "--num_obstacles", "0",
    "--num_resources", "20",
    # "--randomize_num_resources",
    # "--curriculum_num_resources",

    "--observation_radius", "999999",
    # "--observation_mask",
    # "--available_actions_mask",

    "--num_env_steps", "20000000",
    "--episode_length", "400",
    "--max_cycles", "400",
    "--num_mini_batch", "10",

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
    "--hidden_size", "128",
    "--layer_N", "2",

    "--diffusion_model_type", "dit1d",
    "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "100",
    "--cuda",
    "--cuda_idx", "3",

    "--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)