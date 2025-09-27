from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-comms-e200-20x20",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_map_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "2",
    "--num_haulers", "2",
    "--num_prospectors", "2",
    "--num_superbots", "0",

    "--hauler_capacity", "5",

    "--world_no_reset",
    "--world_size", "20",
    "--num_obstacles", "0",
    "--num_resources", "20",

    "--observation_radius", "2",
    "--observation_mask",
    # "--available_actions_mask",

    "--num_env_steps", "30000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "10",

    "--algorithm_class", "qmas.variant_three_modules.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_three_modules.policy.QmasPolicy",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "1024",
    "--layer_N", "3",

    "--diffusion_model_type", "jannerunet",
    "--prediction_ensemble_size", "1",
    "--prediction_history_window", "64",
    # "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "100",
    "--n_training_threads", "32",
    "--cuda",
    "--cuda_idx", "4",
    "--cuda_idx_predictor", "5",

    "--save_interval", "100000",
    "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)