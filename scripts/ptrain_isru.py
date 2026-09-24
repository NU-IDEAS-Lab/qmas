from copy import copy
import multiprocessing
import importlib
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cuda_idx = [0, 1, 5, 7]
scripts = [
    # {
    #     "script": "train_patrolling",
    #     "overrides": {
    #         "algorithm_name": "rmappo",
    #     }
    # },
    # {
    #     "script": "train_patrolling",
    #     "overrides": {
    #         "observation_mask": False,
    #     }
    # },
    {
        "script": "train_isru2"
    },
    {
        "script": "train_isru3",
        "overrides": {
            "cuda_idx_predictor": "4",
        }
    },
    {
        "script": "train_isru5"
    },
    {
        "script": "train_isru6",
        "overrides": {
            "cuda_idx_predictor": "5",
        }
    },
]


# Replace args function.
def argreplace(args, arg_name, arg_value):
    if f"--{arg_name}" in args:
        if type(arg_value) == bool:
            if arg_value:
                if f"--no-{arg_name}" in args:
                    args.remove("--no-" + arg_name)
                args.append(f"--{arg_name}")
            else:
                if f"--{arg_name}" in args:
                    args.remove(f"--{arg_name}")
                args.append("--no-" + arg_name)
        else:
            index = args.index(f"--{arg_name}")
            args[index + 1] = arg_value
    else:
        if type(arg_value) == bool:
            if arg_value:
                args.append(f"--{arg_name}")
            else:
                args.append("--no-" + arg_name)
        else:
            args += [f"--{arg_name}", arg_value]
    return args

threads = []
for i, script in enumerate(scripts):
    module = importlib.import_module(f"{script['script']}")
    args = copy(module.args)
    # args = argreplace(args, "cuda_idx", str(cuda_idx[i]))
    # args = argreplace(args, "state_encoder_output_dim", f"128")
    # args = argreplace(args, "observation_mask", True)

    # args = argreplace(args, "ex_env_communication_mode", "merge_minimum_uq")
    args = argreplace(args, "comm_uq_threshold_temperature", "0.1")

    # args = argreplace(args, "n_rollout_threads", "66")
    # args = argreplace(args, "num_env_steps", "1000000")
    # args = argreplace(args, "num_mini_batch", "30")
    # args = argreplace(args, "use_linear_lr_decay", False)
    # args = argreplace(args, "gnn_neighbor_scoring", True)
    # args = argreplace(args, "gnn_skip_connections", True)
    # args = argreplace(args, "use_gnn_mlp_policy", False)
    # args = argreplace(args, "gnn_layer_N", "2")
    # args = argreplace(args, "gnn_hidden_size", "64")
    # args = argreplace(args, "ppo_epoch", "15")
    # args = argreplace(args, "gnn_use_dense_obs", True)
    # args = argreplace(args, "state_encoder", False)
    # args = argreplace(args, "disable_observation_global", True)
    # args = argreplace(args, "prediction_history_include_actions", True)
    # args = argreplace(args, "algorithm_name", "mappo")

    # args = argreplace(args, "diffusion_model_type", "gnn")
    # args = argreplace(args, "diffusion_sample_steps", "5")
    # args = argreplace(args, "diffusion_autoregression_steps", "7")
    # args = argreplace(args, "gnn_diffusion_depth", "2")
    # args = argreplace(args, "gnn_diffusion_d_model", "32")
    # args = argreplace(args, "gnn_diffusion_temporal_depth", "1")

    # args = argreplace(args, "data_chunk_length", "2")
    # args = argreplace(args, "recurrent_N", "1")

    current_experiment_name = args[args.index("--experiment_name") + 1] if "--experiment_name" in args else "experiment"
    args = argreplace(args, "experiment_name", current_experiment_name.replace("e100", "e150"))
    args = argreplace(args, "episode_length", "150")
    args = argreplace(args, "max_cycles", "150")

    # args = argreplace(args, "use_wandb", False)

    overrides = script.get("overrides", {})
    for key in overrides:
        args = argreplace(args, key, overrides[key])

    p = multiprocessing.Process(target=module.main, args=(args,))
    p.start()
    threads.append(p)

for t in threads:
    t.join()