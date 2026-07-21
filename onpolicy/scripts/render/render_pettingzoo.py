#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket

# third-party packages
import numpy as np
import setproctitle
import torch
import random

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.pettingzoo.Pettingzoo_Env import PettingzooEnv
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, DummyVecEnv, SubprocVecEnv

from onpolicy.scripts.train.train_pettingzoo import parse_args, validateArgs as train_validateArgs, get_environment_class

def validateArgs(all_args):
    ''' Validates the arguments. '''
    train_validateArgs(all_args)

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_render_rollout_threads==1, ("only support to use 1 env to render.")


def make_render_env(all_args):
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

    if all_args.n_render_rollout_threads == 1:
        return single_vecenv_class([get_env_fn(0)])
    else:
        return multi_vecenv_class([get_env_fn(i) for i in range(
            all_args.n_render_rollout_threads)])

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    validateArgs(all_args)

    # Torch configuration.
    # This is equivalent to torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.compiled_autograd = True
    torch.backends.cudnn.benchmark = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{all_args.cuda_idx}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # Set CUBLAS_WORKSPACE_CONFIG for reproducibility:
            # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

            # Set PyTorch flags for reproducibility:
            torch.use_deterministic_algorithms(True)
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir and video dir
    run_dir = Path(all_args.model_dir).parent.absolute()
    if all_args.save_videos and all_args.video_dir == "":
        video_dir = run_dir / "videos"
        all_args.video_dir = str(video_dir)

        if not video_dir.exists():
            os.makedirs(str(video_dir))
        
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
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.pettingzoo_runner import PettingzooRunner as Runner
    else:
        from onpolicy.runner.separated.pettingzoo_runner import PettingzooRunner as Runner

    runner = Runner(config)
    try:
        runner.render()
    except KeyboardInterrupt:
        pass
    
    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
