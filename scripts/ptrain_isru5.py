import os
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_X_DISABLE_VIEWER"] = "true"

from onpolicy.scripts.train.train_pettingzoo import main

args = [
    "--experiment_name", "isru-graphObsGlobalEnc-1prospectorExtractor-1hauler-e200-r5-h5-o5-20x20",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_graph_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "1",
    "--num_haulers", "1",
    "--num_prospectors", "0",
    "--num_superbots", "0",

    "--hauler_capacity", "5",

    # "--world_no_reset",
    "--world_size", "20",
    "--num_obstacles", "0",
    "--num_resources", "5",
    # "--randomize_num_resources",
    # "--curriculum_num_resources",

    "--noisy_memory",

    "--observation_radius", "5",
    # "--observation_mask",
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
    "--prediction_disable",

    "--state_encoder",

    # "--seed", "0",

    "--n_rollout_threads", "50",
    "--threaded_training",
    "--cuda",
    # "--cuda_idx", "5",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

# Run using multiple different seeds in different processes.
import multiprocessing

threads = []
cuda_idx = [5, 5, 7, 7]
for i in range(len(cuda_idx)):
    args_seeded = args + ["--seed", str(i)]
    args_seeded += ["--cuda_idx", str(cuda_idx[i])]
    p = multiprocessing.Process(target=main, args=(args_seeded,))
    p.start()
    threads.append(p)

for t in threads:
    t.join()
