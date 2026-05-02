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
        args += [f"--{arg_name}", arg_value]
    return args

threads = []
for i, script in enumerate(scripts):
    module = importlib.import_module(f"{script}")
    args = module.args
    args = argreplace(args, "cuda_idx", str(cuda_idx[i]))
    args = argreplace(args, "experiment_name", f"BLAH{i}")
    args = argreplace(args, "use_wandb", False)

    p = multiprocessing.Process(target=module.main, args=(args,))
    p.start()
    threads.append(p)

for t in threads:
    t.join()