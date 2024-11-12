import random

from gymnasium.spaces.utils import flatten, flatten_space
from gymnasium.spaces import Dict, Graph
import numpy as np


class PettingzooEnv(object):
    '''Wrapper to make Pettingzoo environments compatible'''

    def __init__(self, environment_class, args):
        self.args = args
        self.num_agents = args.num_agents
        
        self.env = environment_class(
            args=args
        )
        
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.use_obs_instead_of_state = args.use_obs_instead_of_state
        self.state_per_agent = args.state_per_agent

        # Set up action space.
        self.action_space = [self.env.action_spaces[a] for a in self.env.possible_agents]

        # Determine whether observations should be flattened.
        ospace = self.env.observation_spaces[self.env.possible_agents[0]]
        self.flatten_observations = type(ospace) == Dict
        if self.flatten_observations:
            for k, v in ospace.spaces.items():
                if type(v) == Graph:
                    self.flatten_observations = False
                    break

        # Set up observation space.
        if self.flatten_observations:
            self.observation_space = [flatten_space(self.env.observation_spaces[a]) for a in self.env.possible_agents]
        else:
            self.observation_space = [self.env.observation_spaces[a] for a in self.env.possible_agents]
        
        # Set up global observation space.
        self.flatten_observations_global = type(self.env.state_space) == Dict
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
        elif self.state_per_agent:
            ret_share_obs = self._share_obs_wrapper(self.env.state())
        else:
            ret_share_obs = self._share_state_wrapper(self.env.state())
        ret_available_actions = self._available_actions_wrapper(self.env.available_actions)

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
            elif self.state_per_agent:
                ret_share_obs = self._share_obs_wrapper(self.env.state())
            else:
                ret_share_obs = self._share_state_wrapper(self.env.state())
            ret_available_actions = self._available_actions_wrapper(self.env.available_actions)
            info = self._info_wrapper(info)

            # Convert reward.
            rewards += np.array([reward[a] for a in self.env.possible_agents]).reshape(-1, 1)

            # Increase the step count.
            for a in self.env.possible_agents:
                self.deltaSteps[a] += 1

            # Check if any agents are ready.
            # All agents are considered ready if step skipping is disabled.
            ready = not self.args.skip_steps or any([info[a.id]["ready"] for a in self.env.agents])

        self.ppoSteps += 1

        # If we are sharing the reward, then we need to sum the rewards.
        if self.share_reward:
            global_reward = np.sum(rewards)
            rewards = global_reward * np.ones((self.num_agents, 1), dtype=np.float32)
        
        # Convert back from numpy to list.
        rewards = [rewards[i] for i in range(self.num_agents)]

        info["deltaSteps"] = [[self.deltaSteps[a]] for a in self.env.possible_agents]
        info["ready"] = [info[a.id]["ready"] for a in self.env.possible_agents]

        return ret_obs, ret_share_obs, rewards, done, info, ret_available_actions

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _available_actions_wrapper(self, available_actions):
        res = np.array([available_actions[a] for a in self.env.possible_agents])
        return res

    def _obs_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations:
            obs = flatten(self.env.observation_spaces, obs)
            obs = np.reshape(obs, (self.num_agents, -1))
        else:
            obs = [obs[a] for a in self.env.possible_agents]
        
        return obs
    
    def _share_state_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations_global:
            # Flatten the PZ observation.
            res = flatten(self.env.state_space, obs)
        else:
            res = obs
        return res

    def _share_obs_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations_global:
            res = []
            for a in self.env.possible_agents:
                res.append(flatten(self.env.state_space, obs[a]))
            res = np.array(res)
            res = np.reshape(res, (self.num_agents, -1))
        else:
            res = []
            for a in self.env.possible_agents:
                res.append(obs[a])
            res = np.array(res)
        return res


    def _info_wrapper(self, info):
        return info
