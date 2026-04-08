from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-1superBot-e200-r10-h10-o20-20x20-newObs",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_graph_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "0",
    "--num_haulers", "0",
    "--num_prospectors", "0",
    "--num_superbots", "1",

    "--hauler_capacity", "10",

    # "--world_no_reset",
    "--world_size", "20",
    "--num_obstacles", "0",
    "--num_resources", "10",
    # "--randomize_num_resources",
    # "--curriculum_num_resources",

    "--noisy_memory",

    "--observation_radius", "10",
    # "--observation_mask",
    # "--available_actions_mask",

    "--communication_mode", "full",
    "--movement_mode", "velocity",

    # "--share_reward",

    "--num_env_steps", "15000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "10",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_centralized_V",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "128",
    "--layer_N", "3",

	"--use_gnn_policy",
    # "--use_gnn_mlp_policy",
    "--gnn_dropout_rate", "0.5",

    "--diffusion_model_type", "dit1d",
    # "--prediction_disable",

    "--state_encoder",

    # "--seed", "2",

    "--n_rollout_threads", "50",
    "--threaded_training",
    "--cuda",
    # "--cuda_idx", "5",
    # "--cuda_idx_predictor", "7",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)