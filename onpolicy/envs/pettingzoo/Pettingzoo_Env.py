import random

from gymnasium.spaces.utils import flatten, flatten_space
from gymnasium.spaces import Dict, Graph
import numpy as np
from torch_geometric.data import Data

from onpolicy.utils.util import flatten_mask


class PettingzooEnv(object):
    ''' Wrapper to make Pettingzoo environments compatible with the onpolicy algorithms. '''

    def __init__(self, environment_class, args, rank=0):
        self.args = args
        self.num_agents = args.num_agents
        
        if "args" in environment_class.__init__.__code__.co_varnames:
            if rank == 0:
                print("PettingzooEnv: Attempting to pass argparse namespace directly to environment.")
            self.env = environment_class(
                args=args
            )
        else:
            if rank == 0:
                print("PettingzooEnv: Attempting to pass unpacked argparse namespace to environment.")
            args_dict = self._get_matching_arg_dict(environment_class.__init__, vars(args))
            if rank == 0:
                print(f"PettingzooEnv: Passing the following arguments to the environment: {args_dict}")
            self.env = environment_class(
                **args_dict
            )
        
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.use_obs_instead_of_state = args.use_obs_instead_of_state

        # Set up action space.
        self.action_space = [self.env.action_space(a) for a in self.env.possible_agents]

        # Determine whether observations should be flattened.
        ospace = self.env.observation_space(self.env.possible_agents[0])
        self.flatten_observations = type(ospace) == Dict
        if self.flatten_observations:
            for k, v in ospace.spaces.items():
                if type(v) == Graph:
                    self.flatten_observations = False
                    break

        # Set up observation space.
        if self.flatten_observations:
            self.observation_space = [flatten_space(self.env.observation_space(a)) for a in self.env.possible_agents]
        else:
            self.observation_space = [self.env.observation_space(a) for a in self.env.possible_agents]
        
        # Determine whether global observations should be flattened.
        ospace = self.env.state_space
        self.flatten_observations_global = type(ospace) == Dict
        if self.flatten_observations_global:
            for v in ospace.spaces.values():
                if type(v) == Graph:
                    self.flatten_observations_global = False
                    break

        # Set up global observation space.
        if self.flatten_observations_global:
            self.share_observation_space = [flatten_space(self.env.state_space) for a in self.env.possible_agents]
        else:
            self.share_observation_space = [self.env.state_space for a in self.env.possible_agents]


    def reset(self):
        self.ppoSteps = 0
        self.deltaSteps = {a: 0 for a in self.env.possible_agents}
        obs, _  = self.env.reset()

        ret_obs = self._obs_wrapper(obs)
        if self.use_obs_instead_of_state:
            ret_share_obs = self._share_obs_wrapper(obs)
        else:
            ret_share_obs = self._share_state_wrapper(self.env.state())
        available_actions = {a: self.env.available_actions(a) for a in self.env.possible_agents}
        ret_available_actions = self._available_actions_wrapper(available_actions)

        return ret_obs, ret_share_obs, ret_available_actions

    def step(self, action):

        ready = False
        done = []

        # Set up PZ action dictionary.
        actionPz = {a: None for a in self.env.possible_agents}

        # For any agents which are ready, use the new action.
        for i in range(self.num_agents):
            actionPz[self.env.possible_agents[i]] = action[i]
            self.deltaSteps[self.env.possible_agents[i]] = 0
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        while not ready and (not all(done) or done == []):
            # We want to determine if this is the last step when using syncronized step skipping.
            lastStep = self.args.skip_steps and self.ppoSteps >= self.args.episode_length - 1
            
            # Take a step.
            obs, reward, done, trunc, info = self.env.step(actionPz, lastStep=lastStep)

            # Convert the done dict to a list.
            done = [done[a] for a in self.env.possible_agents]
            # Convert the trunc dict to a list.
            trunc = [trunc[a] for a in self.env.possible_agents]

            # Consider the agent done if done OR truncated flags set.
            done = [d or t for d, t in zip(done, trunc)]

            ret_obs = self._obs_wrapper(obs)
            if self.use_obs_instead_of_state:
                ret_share_obs = self._share_obs_wrapper(obs)
            else:
                ret_share_obs = self._share_state_wrapper(self.env.state())
            available_actions = {a: self.env.available_actions(a) for a in self.env.possible_agents}
            ret_available_actions = self._available_actions_wrapper(available_actions)
            info = self._info_wrapper(info)

            # Convert reward.
            rewards += np.array([reward[a] for a in self.env.possible_agents]).reshape(-1, 1)

            # Increase the step count.
            for a in self.env.possible_agents:
                self.deltaSteps[a] += 1

            # Check if any agents are ready.
            # All agents are considered ready if step skipping is disabled.
            def is_ready(a):
                if a in info and "ready" in info[a]:
                    return info[a]["ready"]
                return True
            ready = not self.args.skip_steps or any([is_ready(a) for a in self.env.agents])

        self.ppoSteps += 1

        # If we are sharing the reward, then we need to sum the rewards.
        if self.share_reward:
            global_reward = np.sum(rewards)
            rewards = global_reward * np.ones((self.num_agents, 1), dtype=np.float32)
        
        # Convert back from numpy to list.
        rewards = [rewards[i] for i in range(self.num_agents)]

        info["deltaSteps"] = [[self.deltaSteps[a]] for a in self.env.possible_agents]

        return ret_obs, ret_share_obs, rewards, done, info, ret_available_actions

    def seed(self, seed=None):
        if seed is None:
            seed = 1
        random.seed(seed)
        np.random.seed(seed)

    def close(self):
        self.env.close()

    def _available_actions_wrapper(self, available_actions):
        res = np.array([available_actions[a] for a in self.env.possible_agents])
        return res

    def _obs_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations:
            obs = flatten(self.env.observation_spaces, obs)
            res = np.reshape(obs, (self.num_agents, -1)).astype(np.float32)
        else:
            res = []
            for a in self.env.possible_agents:
                # Check if type of any values in obs is a graph.
                obs_a = obs[a]
                if type(obs_a) == dict:
                    # Ensure dictionary ordering.
                    obs_a = dict(sorted(obs_a.items()))

                    typeSet = set([type(v) for v in obs_a.values()])
                    if Data in typeSet:
                        # If so, we want the observation to be a single-element array of objects.
                        o = np.empty((len(obs_a),), dtype=object)
                        for i, k in enumerate(obs_a.keys()):
                            o[i] = obs_a[k]
                        obs_a = o
                res.append(obs_a)

        
        return res
    
    def _share_state_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations_global:
            # Flatten the PZ observation.
            res = flatten(self.env.state_space, obs).astype(np.float32)
        else:
            res = obs
        return res

    def _share_obs_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations_global:
            res = []
            for a in self.env.possible_agents:
                res.append(flatten(self.env.state_space, obs[a]))
            res = np.array(res, dtype=np.float32)
            res = np.reshape(res, (self.num_agents, -1))
        else:
            res = []
            for a in self.env.possible_agents:
                res.append(obs[a])
            res = np.array(res)
        return res


    def _info_wrapper(self, info):
        ''' Converts the info dictionary to a format that is compatible with the onpolicy algorithms. '''
        def strip_agents_info(info):
            for a in self.env.possible_agents:
                if a in info:
                    del info[a]
            return info
        # Set up new location for the combined visibility mask.
        if "visibility_mask" not in info:
            all_viz = []
            for a in self.env.possible_agents:
                i = info[a]
                if "visibility_mask" in i:
                    viz = i["visibility_mask"]
                    if self.flatten_observations:
                        viz = flatten_mask(self.env.observation_space(a), viz)
                    all_viz.append(viz)
                else:
                    break
            if all_viz:
                # Combine the visibility masks into a single tensor.
                all_viz = np.array(all_viz, dtype=np.float32)
                info["visibility_mask"] = all_viz
        
        # Flatten the state visibility mask if needed.
        if "state_visibility_mask" in info and self.flatten_observations_global:
            svm = info["state_visibility_mask"]
            svm = flatten_mask(self.env.state_space, svm)
            info["state_visibility_mask"] = svm.astype(np.float32)

        # Combine the global observations into a single tensor if needed.
        if "observation_global" not in info:
            all_obs = []
            for a in self.env.possible_agents:
                i = info[a]
                if "observation_global" in i:
                    obs = i["observation_global"]
                    if self.flatten_observations_global:
                        obs = flatten(self.env.state_space, obs)
                    all_obs.append(obs)
                else:
                    break
            if all_obs:
                # Combine the global observations into a single tensor.
                all_obs = np.array(all_obs, dtype=np.float32)
                info["observation_global"] = all_obs
        
        return strip_agents_info(info) 

    def _get_matching_arg_dict(self, fn, args_input):
        ''' Returns a dictionary of arguments that are in both the args_input Namespace and the fn signature. '''
        arg_count = fn.__code__.co_argcount
        args = fn.__code__.co_varnames[:arg_count]

        args_dict = {}
        for k, v in args_input.items():
            if k in args:
                args_dict[k] = v
        return args_dict