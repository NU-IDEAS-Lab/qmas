from onpolicy.scripts.train.train_pettingzoo import main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"
args = [
    "--experiment_name", "hanabi-obsSeer-noObsMasking-rwdInternal-512x2",
    "--project_name", "qmas",
    "--env_class", "isru_zoo.env.Hanabi_pred.parallel_env",
    "--user_name", "ideas-mas",

    "--hanabi_observation_type", "seer",
    # "--observation_mask",
    "--num_env_steps", "10000000000",
    "--episode_length", "100",
    "--max_cycles", "100",
    "--num_mini_batch", "10",

    "--algorithm_class", "qmas.variant_two_modules.new.algorithm.QmasAlgorithm",
    "--policy_class", "qmas.variant_two_modules.new.policy.QmasPolicy",
    "--algorithm_name", "mappo",
    "--use_centralized_V",
    "--use_gae",
    "--share_policy",
    "--use_ReLU",
    "--hidden_size", "512",
    "--layer_N", "2",

    "--lr", "7e-4",
    "--critic_lr", "1e-3",
    "--entropy_coef", "0.015",
    "--ppo_epoch", "15",

    "--diffusion_model_type", "dit1d",
    "--prediction_ensemble_size", "5",
    "--prediction_history_window", "4",
    "--prediction_disable",

    "--seed", "0",

    "--n_rollout_threads", "1000",
    "--n_training_threads", "32",
    "--threaded_training",
    "--cuda",
    "--cuda_idx", "6",
    # "--cuda_idx_predictor", "7",

    "--save_interval", "200000",
    # "--save_checkpoints",
    "--results_dir", "/data/group/mas/qmas/results",
    "--use_wandb",
]

main(args)