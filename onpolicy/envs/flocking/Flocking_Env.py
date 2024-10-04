import random

from flocking_zoo.env.flocking_zoo import parallel_env
from flocking_zoo.env.flocking_graph import FlockingGraph
from flocking_zoo.env.communication_model import CommunicationModel
from gymnasium.spaces.utils import flatten, flatten_space
from gymnasium.spaces import Dict, Graph
import numpy as np


class FlockingEnv(object):
    '''Wrapper to make the Flocking Zoo environment compatible'''

    def __init__(self, args):
        self.args = args
        self.num_agents = args.num_agents
        
        self.env = parallel_env(
            num_shepherds = args.num_shepherds,
            num_sheep = args.num_sheep,
            comms_model = CommunicationModel(
                model = args.communication_model,
                p = args.communication_probability
            ),
            speed_shepherds = args.speed_shepherds,
            speed_sheep = args.speed_sheep,
            alpha = args.alpha,
            beta = args.beta,
            observation_radius = args.observation_radius,
            action_method = args.action_method,
            observe_method = args.observe_method,
            observe_method_global = args.observe_method_global,
            attrition_method = args.attrition_method,
            attrition_random_probability = args.attrition_random_probability,
            attrition_times = args.attrition_fixed_times,
            attrition_min_agents = args.attrition_min_agents,
            reward_method_terminal = args.reward_method_terminal,
            reward_interval = args.reward_interval,
            max_cycles = -1 if self.args.skip_steps_sync or self.args.skip_steps_async else args.max_cycles,
            max_nodes = args.gnn_max_nodes,
            sheep_force_avoid_sheep = args.sheep_force_avoid_sheep,
            sheep_force_avoid_shepherd = args.sheep_force_avoid_shepherd,
            sheep_force_align = args.sheep_force_align,
            sheep_force_cluster = args.sheep_force_cluster,
            sheep_radius_avoid_sheep=args.sheep_radius_avoid_sheep,
            sheep_radius_avoid_shepherd=args.sheep_radius_avoid_shepherd,
            randomize_graph=args.randomize_graph,
            regenerate_graph_on_reset=args.regenerate_graph_on_reset,
            random_graph_danger_zones=args.random_graph_danger_zones,
            random_graph_grid_size=args.random_graph_grid_size,
            random_graph_grid_border=args.random_graph_grid_border,
        )
        
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # Set up action space.
        self.action_space = [self.env.action_spaces[a] for a in self.env.possible_agents]

        # Determine whether observations should be flattened.
        ospace = self.env.observation_spaces[self.env.possible_agents[0]]
        self.flatten_observations = type(ospace) == Dict
        if self.flatten_observations:
            for v in ospace.spaces.values():
                if type(v) == Graph:
                    self.flatten_observations = False
                    break

        # Set up observation space.
        if self.flatten_observations:
            self.observation_space = [flatten_space(self.env.observation_spaces[a]) for a in self.env.possible_agents]
        else:
            self.observation_space = [self.env.observation_spaces[a] for a in self.env.possible_agents]
        
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
        self.prevAction = {a: None for a in self.env.possible_agents}
        self.deltaSteps = {a: 0 for a in self.env.possible_agents}
        obs, _  = self.env.reset()

        combined_obs = {
            "obs": self._obs_wrapper(obs),
            "share_obs": self._share_obs_wrapper(self.env.state_all())
        }

        return combined_obs

    def step(self, action_metadata):

        ready = False
        done = []
        if type(action_metadata) == dict:
            action = action_metadata["action"]
            last_step = False
            if "metadata" in action_metadata:
                metadata = action_metadata["metadata"]
                last_step = metadata["last_step"]
        else:
            action = action_metadata
            last_step = False


        # Start with the previous action.
        actionPz = self.prevAction

        # For any agents which are ready, use the new action.
        for i in range(self.num_agents):
            if np.all(action[i]) != None:
                actionPz[self.env.possible_agents[i]] = action[i]

                # Reset step count.
                self.deltaSteps[self.env.possible_agents[i]] = 0
            elif not self.args.skip_steps_async:
                raise ValueError(f"Action cannot be None when skip_steps_async is False. Agent: {i}")

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        while not ready and (not all(done) or done == []):
            # We want to determine if this is the last step when using syncronized step skipping.
            lastStep = last_step or (self.args.skip_steps_sync and self.ppoSteps >= self.args.episode_length - 1)
            
            # Take a step.
            obs, reward, done, trunc, info = self.env.step(actionPz, lastStep=lastStep)

            # Convert the done dict to a list.
            done = [done[a] for a in self.env.possible_agents]
            # Convert the trunc dict to a list.
            trunc = [trunc[a] for a in self.env.possible_agents]

            # Consider the agent done if done OR truncated flags set.
            done = [d or t for d, t in zip(done, trunc)]

            combined_obs = {
                "obs": self._obs_wrapper(obs),
                "share_obs": self._share_obs_wrapper(self.env.state_all())
            }

            # Increase reward.
            rewards += np.array([reward[a] for a in self.env.possible_agents]).reshape(-1, 1)

            info = self._info_wrapper(info)

            # Increase the step count.
            for a in self.env.possible_agents:
                self.deltaSteps[a] += 1

            # Only run once if skip_steps_sync is false.
            if not self.args.skip_steps_sync:
                break

            # Check if any agents are ready
            ready = any([info[a.id]["ready"] for a in self.env.agents])

        # If we are sharing the reward, then we need to sum the rewards.
        if self.share_reward:
            global_reward = np.sum(rewards)
            rewards = global_reward * np.ones((self.num_agents, 1), dtype=np.float32)
        
        # Convert back from numpy to list.
        rewards = [rewards[i] for i in range(self.num_agents)]

        info["deltaSteps"] = [[self.deltaSteps[a]] for a in self.env.possible_agents]
        info["ready"] = [info[a.id]["ready"] for a in self.env.possible_agents]

        self.ppoSteps += 1

        # Update the previous action.
        self.prevAction = actionPz

        return combined_obs, rewards, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations:
            obs = flatten(self.env.observation_spaces, obs)
            obs = np.reshape(obs, (self.num_agents, -1))
        else:
            obs = [obs[a] for a in self.env.possible_agents]
        
        return obs
    
    def _share_obs_wrapper(self, obs):

        # Flatten the PZ observation.
        if self.flatten_observations_global:
            res = []
            for a in self.env.possible_agents:
                res.append(flatten(self.env.state_space, obs[a]))
            res = np.array(res)
            res = np.reshape(res, (self.num_agents, -1))
            return res

            #This older code below is for use with the state() method. Above code is for the state_all() method.
            # Flatten the PZ observation.
            obs = flatten(self.env.state_space, obs)
        else:
            res = []
            for a in self.env.possible_agents:
                res.append(obs[a])
            res = np.array(res)
            return res
        
            #This older code below is for use with the state() method. Above code is for the state_all() method.
            # obs = obs

        return obs

    def _info_wrapper(self, info):
        return info
