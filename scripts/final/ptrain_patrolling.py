import multiprocessing
import importlib

cuda_idx = [0, 1, 2, 3, 4, 5, 6, 7]

scripts = [
    "train_patrolling",
    "train_patrolling_pred",
    "train_patrolling_pred_uq",
    "train_patrolling_comms_pred",
    "train_patrolling_comms_pred_uq",
]

# Replace args function.
def argreplace(args, arg_name, arg_value):
    if f"--{arg_name}" in args:
        if type(arg_value) == bool:
            if arg_value:
                args.remove("--no-" + arg_name)
                args.append(f"--{arg_name}")
            else:
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
    args = argreplace(args, "observation_mask", True)
    args = argreplace(args, "n_rollout_threads", "33")
    args = argreplace(args, "num_mini_batch", "10")
    args = argreplace(args, "beta", "5.0")
    args = argreplace(args, "alpha", "0.0")
    args = argreplace(args, "episode_fraction_start_policy", "0.1")
    args = argreplace(args, "use_linear_lr_decay", False)

    # TEST WITH SMALL GRAPH
    args = argreplace(args, "experiment_name", f"test_nodes10_{script}")
    args = argreplace(args, "graph_random_nodes", "10")
    args = argreplace(args, "num_agents", "2")
    args = argreplace(args, "episode_length", "100")
    args = argreplace(args, "max_cycles", "100")

    p = multiprocessing.Process(target=module.main, args=(args,))
    p.start()
    threads.append(p)

for t in threads:
    t.join()