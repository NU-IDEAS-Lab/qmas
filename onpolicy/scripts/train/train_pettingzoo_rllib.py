
import importlib
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.utils import parallel_to_aec

# Use argument parsing and env class logic from train_pettingzoo.py
from onpolicy.config import get_config
from onpolicy.scripts.train.train_pettingzoo import parse_args, get_environment_class

def get_env(env_cls, all_args):
    def env_creator(config=None):
        env = env_cls(args=all_args) if 'args' in env_cls.__init__.__code__.co_varnames else env_cls()
        return parallel_to_aec(env)
    return env_creator


def main():
    parser = get_config()
    # Do not add env_module/env_class here; use parse_args from train_pettingzoo.py
    import sys
    args = parse_args(sys.argv[1:], parser)

    env_cls = get_environment_class(args)
    env_creator = get_env(env_cls, args)
    tune.register_env('custom_pz_env', lambda config: env_creator())

    # Map config values from args to RLLib config
    num_workers = getattr(args, 'n_rollout_threads', 1)
    num_gpus = int(getattr(args, 'cuda', False))
    stop_iters = getattr(args, 'stop_iters', 100)

    config = (
        PPOConfig()
        .environment('custom_pz_env')
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus)
    )

    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(stop={"training_iteration": stop_iters}),
    ).fit()

    print("Best checkpoint:", results.get_best_result().checkpoint)

if __name__ == "__main__":
    main()
