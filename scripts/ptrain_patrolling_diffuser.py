from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    # "--experiment_name", "diffuser-patrolling-obs200-uqEnsemble3-pygNS-commsRL0.5-injectionAppend",
    # "--experiment_name", "diffuser-patrolling-obs200-uqEnsemble3-pygNS-injectionAppend",
    "--project_name", "qmas",
    "--env_class", "patrolling_zoo.patrolling_zoo_v0.parallel_env",
    "--user_name", "ideas-mas",

    "--num_agents", "4",
    "--agent_speed", "40.0",
    "--action_method", "neighbors_with_comm_boolean",
    "--observe_method", "pyg",
    "--observe_method_global", "adjacency",
    "--observation_radius", "200.0",
    "--communication_model", "none",
    "--communication_probability", "0.0",
    "--alpha", "1.0",
    "--beta", "0.5",
    # "--reward_comms_penalty_weight", "0.5",
    "--reward_method_terminal", "average",

    "--graph_name", "milwaukee",
    "--graph_file", "../patrolling_zoo/graphs/milwaukee.graph",
    "--num_env_steps", "1000000",
    "--episode_length", "200",
    "--max_cycles", "200",
    "--num_mini_batch", "10",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "rmappo",
    "--use_gnn_policy",
    # "--use_gnn_mlp_policy",
    "--gnn_layer_N", "10",
    "--gnn_hidden_size", "128",
    "--gnn_skip_connections",
    "--gnn_neighbor_scoring",
    # "--use_recurrent_policy",
    "--use_centralized_V",
    "--use_gae",
    "--use_gae_amadm",
    "--skip_steps",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "512",
    "--layer_N", "4",
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

    "--n_rollout_threads", "33",
    "--cuda",
    # "--cuda_idx", "6",
    # "--cuda_idx_predictor", "4",
    "--threaded_training",
    "--n_training_threads", "16",

    "--save_interval", "10000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

# Run using multiple different seeds in different processes.
import multiprocessing
from copy import copy

threads = []
cuda_idx = [2, 5, 6, 7]

penalty_weights = [0.1, 0.25, 0.5, 1.0]

for i in range(len(cuda_idx)):
    args_modified = copy(args)
    args_modified += ["--reward_comms_penalty_weight", str(penalty_weights[i]),]
    args_modified += ["--experiment_name", f"diffuser-patrolling-obs200-uqEnsemble3-pygNS-commsRL{penalty_weights[i]}-injectionAppend-predObs",]

    args_modified += ["--cuda_idx", str(cuda_idx[i])]
    p = multiprocessing.Process(target=main, args=(args_modified,))
    p.start()
    threads.append(p)

for t in threads:
    t.join()
