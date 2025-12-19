from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-graphObsGlobalEnc-3superBot-e50-r5-h5-20x20-128x3-commsFull",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_graph_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "0",
    "--num_haulers", "0",
    "--num_prospectors", "0",
    "--num_superbots", "3",

    "--hauler_capacity", "5",

    # "--world_no_reset", 
    "--world_size", "20",
    "--num_obstacles", "0",
    "--num_resources", "5",
    # "--randomize_num_resources",
    # "--curriculum_num_resources",

    "--noisy_memory",

    "--observation_radius", "3",
    # "--observation_mask",
    # "--available_actions_mask",

    "--communication_mode", "full",

    "--num_env_steps", "5000000",
    "--episode_length", "50",
    "--max_cycles", "50",
    "--num_mini_batch", "1",

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
    "--use_gnn_mlp_policy",

    "--diffusion_model_type", "dit1d",
    "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "50",
    "--cuda",
    "--cuda_idx", "4",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)