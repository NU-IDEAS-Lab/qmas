#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket
import importlib

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb
import random

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.pettingzoo.Pettingzoo_Env import PettingzooEnv
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, SubprocVecEnv, DummyVecEnv


def get_environment_module(all_args):
    ''' Dynamically imports correct environment class. '''

    module_name, _ = all_args.env_class.rsplit(".", 1)
    try:
        env_module = importlib.import_module(module_name) 
    except ImportError:
        raise ValueError(f"Invalid environment module: {module_name}. Import failed.")
    return env_module

def get_environment_class(all_args):
    ''' Dynamically imports correct environment class. '''

    module_name, class_name = all_args.env_class.rsplit(".", 1)
    try:
        env_module = importlib.import_module(module_name)
        env_class = getattr(env_module, class_name)
    except ImportError:
        raise ValueError(f"Invalid environment module: {module_name}. Import failed.")
    except AttributeError:
        raise ValueError(f"Invalid environment class: {all_args.env_class}. Import failed.")
    return env_class


def add_env_args(parsed_args, parser):
    ''' Parses environment-specific arguments. '''

    env_module = get_environment_module(parsed_args)
    if hasattr(env_module, "add_args"):
        env_module.add_args(parser)
        print(f"Pettingzoo environment {parsed_args.env_class} has additional arguments.")


def validate_env_args(parsed_args):
    ''' Parses environment-specific arguments. '''

    env_module = get_environment_module(parsed_args)
    if hasattr(env_module, "validate_args"):
        print(f"Pettingzoo environment {parsed_args.env_class} has additional arguments validation function.")
        return env_module.validate_args(parsed_args)


def make_train_env(all_args):
    ''' Builds training environments for all threads. '''

    env_class = get_environment_class(all_args)

    def get_env_fn(rank):
        def init_env():
            env = PettingzooEnv(env_class, all_args, rank)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.use_obs_instead_of_state:
        single_vecenv_class = DummyVecEnv
        multi_vecenv_class = SubprocVecEnv
    else:
        single_vecenv_class = ShareDummyVecEnv
        multi_vecenv_class = ShareSubprocVecEnv

    if all_args.n_rollout_threads == 1:
        return single_vecenv_class([get_env_fn(0)])
    else:
        return multi_vecenv_class([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    ''' Builds evaluation environments for all threads. '''

    env_class = get_environment_class(all_args)

    def get_env_fn(rank):
        def init_env():
            env = PettingzooEnv(env_class, all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.use_obs_instead_of_state:
        single_vecenv_class = DummyVecEnv
        multi_vecenv_class = SubprocVecEnv
    else:
        single_vecenv_class = ShareDummyVecEnv
        multi_vecenv_class = ShareSubprocVecEnv

    if all_args.n_rollout_threads == 1:
        return single_vecenv_class([get_env_fn(0)])
    else:
        return multi_vecenv_class([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    ''' Parses Pettingzoo-specific arguments. '''

    import argparse
    
    parser.add_argument("--env_class", type=str, default='', help="specify the environment class")
    parser.add_argument("--max_cycles", type=int, default=1000,
                        help="max number of cycles for the environment.")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="number of controlled players.")
    parser.add_argument("--skip_steps", action=argparse.BooleanOptionalAction, 
                    default=False, 
                    help="by default False. If True, skips steps for which no agents are ready to take an action.")
    parser.add_argument("--eval_deterministic", action=argparse.BooleanOptionalAction, 
                        default=True, 
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action=argparse.BooleanOptionalAction, 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")
    parser.add_argument("--cuda_idx", type=int, default=0, 
                        help="Index of the GPU to use")
    
    # Parse once to get the environment name.
    parsed_args, unknown_args = parser.parse_known_args(args)
    add_env_args(parsed_args, parser)

    # Parse again to get environment-specific arguments.
    parsed_args, unknown_args = parser.parse_known_args(unknown_args, namespace=parsed_args)

    if unknown_args:
        import warnings
        warnings.warn(f"Unknown arguments: {unknown_args}")

    return parsed_args


def validateArgs(all_args):
    ''' Validates Pettingzoo-specific arguments. '''

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise ValueError(f"Algorithm name {all_args.algorithm_name} not recognized.")
    
    # Set max_cycles to -1 if skip_steps is set.
    if all_args.skip_steps:
        all_args.max_cycles = -1

    # Create an environment object to validate it.
    env_class = get_environment_class(all_args)
    env = env_class() # use default arguments

    # Set the environment name if it is not set.
    if all_args.env_name == "" and hasattr(env, "metadata") and "name" in env.metadata:
        all_args.env_name = env.metadata["name"]

    # Check whether the environment has a callable state function.
    if not all_args.use_obs_instead_of_state:
        if not hasattr(env, "state") or not callable(env.state):
            raise ValueError(f"Environment class {env_class} does not have state function, but use_obs_instead_of_state is set false.")

    print("Pettingzoo arguments validated: base")

    # Validate environment arguments.
    validate_env_args(all_args)

    print(f"Pettingzoo arguments validated: {all_args.env_class}")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    validateArgs(all_args)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.cuda_idx}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    base_dir = all_args.results_dir
    if base_dir == "":
        base_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results"
    run_dir = Path(base_dir) / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # get date and time in format 20230816-150432
    import datetime
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                            project=all_args.project_name,
                            entity=all_args.user_name,
                            notes=socket.gethostname(),
                            name="-".join([
                                all_args.algorithm_name,
                                all_args.experiment_name,
                                str(date_time),
                                "seed" + str(all_args.seed)
                            ]),
                            group=all_args.env_name,
                            dir=str(run_dir),
                            job_type="training",
                            reinit=True)
        run_dir = Path(wandb.run.dir)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    print(f"run_dir: {run_dir}")

    setproctitle.setproctitle("-".join([
        all_args.project_name, 
        all_args.env_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.pettingzoo_runner import PettingzooRunner as Runner
    else:
        raise NotImplementedError("Pettingzoo wrapper does not yet support separate policies.")
        from onpolicy.runner.separated.pettingzoo_runner import PettingzooRunner as Runner

    try:
        runner = Runner(config)
        runner.run()
    except KeyboardInterrupt:
        wandb.finish(exit_code=1)
        print("exit due to keyboard interrupt")

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # Save the models.
    runner.save()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
