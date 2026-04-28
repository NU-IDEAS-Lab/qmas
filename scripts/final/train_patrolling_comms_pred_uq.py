from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "patrolling-randomOR-comms-pred-uqstd-draft1",
    "--project_name", "qmas",
    "--env_class", "patrolling_zoo.patrolling_zoo_v0.parallel_env",
    "--user_name", "ideas-mas",

    "--num_agents", "4",
    "--agent_speed", "40.0",
    "--action_method", "neighbors_with_comm_boolean",
    "--observe_method", "pyg",
    "--observe_method_global", "coordinates",
    "--observation_radius", "200.0",
    "--observation_radius_random_min", "50.0",
    "--observation_radius_random_max", "300.0",
    "--communication_model", "none",
    "--communication_probability", "0.0",
    "--alpha", "1.0",
    "--beta", "0.1",
    "--reward_comms_penalty_weight", "0.1",
    "--reward_method_terminal", "none",
    "--reward_interval", "1",

    # "--graph_name", "milwaukee",
    # "--graph_file", "../patrolling_zoo/graphs/milwaukee.graph",
    "--graph_random",
    "--regenerate_graph_on_reset",
    "--regenerate_graph_every", "1",

    "--num_env_steps", "1000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "10",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "rmappo",
    "--use_gnn_policy",
    # "--use_gnn_mlp_policy",
    "--gnn_layer_N", "5",
    "--gnn_hidden_size", "32",
    "--gnn_skip_connections",
    "--gnn_neighbor_scoring",
    # "--use_recurrent_policy",
    "--use_centralized_V",
    "--use_gae",
    "--use_gae_amadm",
    "--skip_steps",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "64",
    # "--critic_hidden_size", "512",
    "--layer_N", "3",
    "--use_linear_lr_decay",
    "--ppo_epoch", "5",
    "--seed", "0",

    "--diffusion_model_type", "dit1d",
    # "--prediction_history_window", "16",
    "--prediction_during_training",
    "--prediction_uq_method", "ensemble",
    "--prediction_uq_injection_method", "append",
    # "--prediction_disable",
    "--prediction_ensemble_size", "3",
    "--state_encoder",

    # "--episode_fraction_start_prediction", "0.2",

    "--n_rollout_threads", "66",
    "--cuda",
    "--cuda_idx", "2",
    # "--cuda_idx_predictor", "6",
    "--threaded_training",
    "--n_training_threads", "16",

    "--save_interval", "100000",
    "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)