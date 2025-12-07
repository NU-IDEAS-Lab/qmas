from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-partialObs-1superBot-e200-r5-h5-5x5-128x3",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_partial_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "0",
    "--num_haulers", "0",
    "--num_prospectors", "0",
    "--num_superbots", "1",

    "--hauler_capacity", "5",

    "--world_no_reset",
    "--world_size", "5",
    "--num_obstacles", "0",
    "--num_resources", "5",
    # "--randomize_num_resources",
    # "--curriculum_num_resources",

    "--observation_radius", "2",
    "--observation_mask",
    # "--available_actions_mask",

    "--num_env_steps", "5000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "2",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_centralized_V",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "128",
    "--layer_N", "3",

    "--diffusion_model_type", "dit1d",
    "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "25",
    "--cuda",
    "--cuda_idx", "5",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)