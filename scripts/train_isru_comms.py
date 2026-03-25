from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-comms-e200-20x20-4superBots",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_map_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "0",
    "--num_haulers", "0",
    "--num_prospectors", "0",
    "--num_superbots", "4",

    "--hauler_capacity", "5",

    # "--world_no_reset",
    "--world_size", "20",
    "--num_obstacles", "0",
    "--num_resources", "20",
    # "--noisy_memory",

    "--observation_radius", "3",
    # "--observation_mask",
    # "--available_actions_mask",

    "--num_env_steps", "10000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "4",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.comms_heuristic.policy.QmasPolicy",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "1024",
    "--layer_N", "3",

    "--diffusion_model_type", "dit1d",
    "--prediction_ensemble_size", "1",
    "--prediction_history_window", "8",
    # "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "1",
    "--n_training_threads", "16",
    "--cuda",
    "--cuda_idx", "1",
    "--threaded_training",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    # "--use_wandb",
]

main(args)