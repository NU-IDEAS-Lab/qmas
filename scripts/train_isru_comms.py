from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

args = [
    "--experiment_name", "isru-heuristic-dry-run",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.isru_v0.parallel_env_map_obs",
    "--user_name", "ideas-mas",

	"--num_extractors", "2",
    "--num_haulers", "2",
    "--num_prospectors", "2",
    "--num_superbots", "0",

    "--hauler_capacity", "10",

    "--world_no_reset",
    "--world_size", "10",
    "--num_obstacles", "0",
    "--num_resources", "10",

    "--observation_radius", "999999",
    # "--observation_mask",
    # "--available_actions_mask",

    "--num_env_steps", "100",
    "--episode_length", "100",
    "--max_cycles", "100",
    "--num_mini_batch", "1",

    "--algorithm_class", "qmas.variant_three_modules.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_three_modules.policy.QmasPolicy",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "1024",
    "--layer_N", "3",

    "--diffusion_model_type", "dit1d",
    "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "1",
    "--cuda",
    "--cuda_idx", "3",

    "--save_interval", "1000",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)