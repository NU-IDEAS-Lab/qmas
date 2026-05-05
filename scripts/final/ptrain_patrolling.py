import multiprocessing
import importlib

cuda_idx = [0, 1, 2, 3, 4, 5, 6, 7]
# cuda_idx = [5, 5, 5, 0, 0]
scripts = [
    "train_patrolling",
    "train_patrolling_pred",
    "train_patrolling_pred_uq",
    "train_patrolling_comms_pred",
    "train_patrolling_comms_pred_uq",
]

# cuda_idx = [5, 6, 7]
# scripts = [
#     "train_patrolling",
#     "train_patrolling_pred",
#     "train_patrolling_comms_pred_uq",
# ]

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
    module = importlib.import_module(f"{script}")
    args = module.args
    args = argreplace(args, "cuda_idx", str(cuda_idx[i]))
    args = argreplace(args, "state_encoder_output_dim", f"128")
    args = argreplace(args, "observation_mask", False)
    args = argreplace(args, "n_rollout_threads", "66")
    args = argreplace(args, "num_mini_batch", "10")
    args = argreplace(args, "beta", "0.5")
    args = argreplace(args, "alpha", "1.0")
    args = argreplace(args, "reward_comms_penalty_weight", "0.0")
    args = argreplace(args, "reward_method_terminal", "average")
    args = argreplace(args, "use_linear_lr_decay", False)
    args = argreplace(args, "reward_interval", "-1")
    args = argreplace(args, "gnn_neighbor_scoring", True)
    args = argreplace(args, "gnn_skip_connections", False)
    args = argreplace(args, "gnn_use_dense_obs", True)
    args = argreplace(args, "state_encoder", False)


    # TEST WITH SMALL GRAPH
    args = argreplace(args, "experiment_name", f"test-nodes10-denseGNN-{script}")
    args = argreplace(args, "graph_random_nodes", "10")
    args = argreplace(args, "regenerate_graph_on_reset", False)
    args = argreplace(args, "num_agents", "2")
    args = argreplace(args, "episode_length", "100")
    args = argreplace(args, "max_cycles", "100")

    p = multiprocessing.Process(target=module.main, args=(args,))
    p.start()
    threads.append(p)

for t in threads:
    t.join()