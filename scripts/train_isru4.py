from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-2e2h1p-e200-r10-h10-o20-20x20",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_graph_obs",
    "--user_name", "ideas-mas",

    "--num_extractors", "2",
    "--num_haulers", "2",
    "--num_prospectors", "1",
    "--num_superbots", "0",

    "--hauler_capacity", "10",

    # "--world_no_reset",
    "--world_size", "20",
    "--num_obstacles", "0",
    "--num_resources", "10",
    # "--randomize_num_resources",
    # "--curriculum_num_resources",

    # "--noisy_memory",

    "--observation_radius", "3",
    "--observation_mask",
    # "--available_actions_mask",

    "--communication_mode", "full",
    "--movement_mode", "velocity",

    "--num_env_steps", "15000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "10",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_gnn_policy",
    "--gnn_hidden_size", "64",
    "--use_centralized_V",
    "--use_gae",
    "--share_policy",

    "--use_ReLU",
    "--hidden_size", "128",
    "--layer_N", "3",

	"--use_gnn_policy",
    # "--use_gnn_mlp_policy",
    # "--gnn_dropout_rate", "0.5",

    "--diffusion_model_type", "gnn",
    "--diffusion_sample_steps", "5",
    "--diffusion_autoregression_steps", "7",
    "--gnn_diffusion_depth", "2",
    "--gnn_diffusion_d_model", "32",
    "--gnn_diffusion_temporal_depth", "1",
    "--prediction_during_training",
    "--prediction_uq_method", "ensemble",
    "--prediction_uq_injection_method", "append",
    # "--prediction_disable",
    "--prediction_ensemble_size", "3",

    # "--state_encoder",
    "--disable_observation_global",
    "--gnn_use_dense_obs",

    "--seed", "0",

    "--n_rollout_threads", "201",
    "--threaded_training",
    "--n_training_threads", "16",
    "--cuda",
    "--cuda_idx", "5",
    "--cuda_idx_predictor", "6",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

if __name__ == "__main__":
    main(args)