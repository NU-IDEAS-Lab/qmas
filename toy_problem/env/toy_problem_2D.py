from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import os
from gymnasium import spaces
import random
import numpy as np
import math
from copy import deepcopy
from matplotlib import pyplot as plt
import networkx as nx
from copy import copy
from enum import IntEnum
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data


def add_args(parser):
    ''' Adds environment arguments. '''
    pass

def parse_args(args):
    ''' Parses environment arguments. '''
    pass


def validate_args(parsed_args):
    ''' Validates the arguments. '''
    pass


def env(*args, **kwargs):
    ''' Returns the environment class. '''
    return parallel_env(*args, **kwargs)


def raw_env(*args, **kwargs):
    ''' Returns the raw environment class. '''
    env = parallel_env(*args, **kwargs)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {
        "name": "toy_problem_v0",
        "render_modes": ["human", "rgb_array"],
        "null_value": -1.0
    }


    def __init__(self,
                 num_agents = 3,
                 max_cycles: int = -1,
                ):
        """
        Initialize the environment.
        """
        super().__init__()

        # Configuration.
        self.max_cycles = max_cycles

        # Create the agents.
        self.possible_agents = list(range(num_agents))

        # Create the action space.
        action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore
        
        # Create the observation space.
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + num_agents*2, ), dtype=np.float32)
        self.observation_spaces = spaces.Dict({agent: obs_space for agent in self.possible_agents}) # type: ignore

        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        self.state_space = obs_space

        self.reset_count = 0
        self.reset()


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        self.reset_count += 1

        if seed != None:
            random.seed(seed)

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        self.agent_states = {agent: np.array([0.0, 0.0], dtype=np.float32) for agent in self.agents}
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)
        self.reference_state = np.array([0.0, 0.0], dtype=np.float32)
        
        # Initialize alpha (velocity) and save base value for variations
        self.alpha = np.array([random.random(), random.random()], dtype=np.float32)
        self.base_alpha = self.alpha.copy()  # Store the initial alpha for reference
        self.alpha_history = [self.alpha.copy()]
        
        self.state_history = {agent: [np.array([0.0, 0.0], dtype=np.float32)] for agent in self.agents}
        self.reference_state_history = [np.array([0.0, 0.0], dtype=np.float32)]

        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.agents}

        # Return the initial observation.
        observation = {agent: self.observe(agent) for agent in self.agents}
        info = {
            agent: {
                "ready": True
            } for agent in self.agents
        }

        return observation, info


    def render(self, pred, figsize=(9, 6), history_length=2):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches.
                
            Returns:
                None
        '''

        # Plot as a line graph using matplotlib.
        plt.figure(figsize=figsize)
        for agent in self.agents:
            print("agent sate hist: ", self.state_history[agent])
            x_hist = [point[0] for point in self.state_history[agent]]
            y_hist = [point[1] for point in self.state_history[agent]]
            plt.plot(x_hist, y_hist, label=f"Follower {agent} (Actual)", color="red", alpha=0.7)
        
        print("reference sate hist: ", self.reference_state_history)
        x_ref = [point[0] for point in self.reference_state_history]
        y_ref = [point[1] for point in self.reference_state_history]
        plt.plot(x_ref, y_ref, label="Leader (Actual)", color="orange", alpha=0.7)

        print("pred shape", pred.shape)
        print("pred: ", pred)
        print("pred[0]: ", pred[0])
                
        colors = ["orange", "green", "red", "yellow", "blue", "purple", "black", "grey"]
        labels = [f"Leader (Predicted)", "Leader Speed (Predicted)", f"Follower 0 (Predicted)"]
        # Leader predicted
        leader_x = []
        leader_y = []
        for i in range(len(pred)):
            leader_x.append(pred[i][0])
            leader_y.append(pred[i][1])
        plt.plot(leader_x, leader_y, color = colors[0], alpha = 1.0, linestyle = "dashed", label = labels[0])
        
        
        # Follower predicted
        follower_x = []
        follower_y = []
        for i in range(len(pred)):
            follower_x.append(pred[i][4])
            follower_y.append(pred[i][5])
        plt.plot(follower_x, follower_y, color = colors[2], alpha = 1.0, linestyle = "dashed", label = labels[2])
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.show()


        # Plot Speed
        plt.figure(figsize=figsize)
        # Leader Speed predicted 
        leader_vx = []
        leader_vy = []
        for i in range(len(pred)):
            leader_vx.append(pred[i][2])
            leader_vy.append(pred[i][3])
        plt.plot(leader_vx, leader_vy, color = colors[1], alpha = 1.0, linestyle = "dashed", label = labels[1])
        
        # Plot alpha history instead of just current alpha
        alpha_x = [alpha[0] for alpha in self.alpha_history]
        alpha_y = [alpha[1] for alpha in self.alpha_history]
        plt.plot(alpha_x, alpha_y, color='g', alpha=0.7, label="Leader Speed (Actual)")
        
        plt.xlabel("X Speed")
        plt.ylabel("Y Speed")
        plt.legend()
        plt.show()
        
        print(f"Leader Actual: {self.reference_state_history}")
        print(f"Leader Predicted: {pred[:, 1]}")

    
    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''
        return self.observation_spaces[agent]


    def action_space(self, agent):
        ''' Returns the action space for the given agent. '''
        return self.action_spaces[agent]


    def available_actions_space(self, agent):
        ''' Generate a Space for the available actions, given the action space. '''

        action_space = self.action_space(agent)
        def get_available_action_space(action_space):
            if action_space.__class__.__name__ in ["Tuple", "Dict"]:
                return spaces.Dict({k: get_available_action_space(v) for k, v in action_space.spaces.items()})
            elif action_space.__class__.__name__ == "Discrete":
                return spaces.MultiBinary(action_space.n)
            elif action_space.__class__.__name__ == "MultiDiscrete":
                return spaces.MultiBinary(len(action_space.nvec), np.max(action_space.nvec))
            else:
                raise NotImplementedError(f"Action space {action_space} not supported for action masking.")
        return get_available_action_space(action_space)


    def state_old(self):
        ''' Returns the global state of the environment.
            This is useful for centralized training, decentralized execution. '''
        
        return self._populateStateSpace(self.possible_agents[0])

    def state(self):
        ''' Similar to the state_old() method, but this returns a customized copy of the state space for each agent.
            This is useful for centralized training, decentralized execution. '''
        
        state = {}
        for agent in self.possible_agents:
            state[agent] = self._populateStateSpace(agent)
        return state


    def observe(self, agent, radius=None, allow_done_agents=False):
        ''' Returns the observation for the given agent.'''

        return self._populateStateSpace(agent)


    def available_actions(self, agent):
        ''' Returns the dictionary of available actions for all agents.
            This is not standard in the Pettingzoo API but is useful. '''
        return self.available_actions_dict[agent]


    def _populateStateSpace(self, agent):
        ''' Returns a populated state/observation space.'''

        obs = np.array([self.reference_state, self.alpha] + [self.agent_states[a] for a in self.possible_agents], dtype=np.float32)
        obs = obs.flatten()
        return obs
    
    def _update_alpha(self):
        """
        Update alpha based on the step count to create a more complex pattern.
        This function creates a time-varying alpha that follows different patterns.
        """
        # Base frequency for oscillation
        freq = 0.1
        angle = freq * self.step_count
        
        # Create a more interesting pattern with multiple frequencies
        self.alpha[0] = self.base_alpha[0] * (math.sin(angle) + 0.5 * math.sin(2.5 * angle))
        self.alpha[1] = self.base_alpha[1] * (math.cos(angle) + 0.5 * math.cos(3.0 * angle))
        
        # Add some random noise to make the trajectory more natural
        noise_magnitude = 0.05 * min(1.0, self.step_count / 50.0)  # Gradually increase noise
        self.alpha += np.random.normal(0, noise_magnitude, size=2)
        
        # Record alpha history
        self.alpha_history.append(self.alpha.copy())
        
    def step(self, action_dict={}, lastStep=False):
        ''''
        Perform a step in the environment based on the given action dictionary.

        Args:
            action_dict (dict): A dictionary containing actions for each agent.

        Returns:
            obs_dict (dict): A dictionary containing the observations for each agent.
            reward_dict (dict): A dictionary containing the rewards for each agent.
            done_dict (dict): A dictionary indicating whether each agent is done.
            info_dict (dict): A dictionary containing additional information for each agent.
        '''
        self.step_count += 1
        
        # Update alpha with time-varying pattern
        self._update_alpha()
        
        obs_dict = {}
        reward_dict = {agent: 0.0 for agent in self.possible_agents}
        truncated_dict = {agent: False for agent in self.possible_agents}
        info_dict = {
            agent: {
                "ready": True
            } for agent in self.possible_agents
        }

        # Perform actions.
        for agent in self.agents:
            if agent in action_dict:
                action = action_dict[agent]

                # Increment the agent state.                
                self.agent_states[agent] += action

                self.state_history[agent].append(self.agent_states[agent])

        # Increment the reference state using the time-varying alpha
        self.reference_state += self.alpha
        self.reference_state_history.append(self.reference_state)

        # Provide reward based on distance to reference state
        for agent in self.agents:
            distance = np.linalg.norm(self.agent_states[agent] - self.reference_state)
            reward_dict[agent] = -distance

        # Perform observations.
        for agent in self.possible_agents:
            agent_observation = self.observe(agent)
            obs_dict[agent] = agent_observation
            info_dict[f"distance/{agent}"] = np.linalg.norm(self.agent_states[agent] - self.reference_state)
            info_dict[f"position/{agent}"] = self.agent_states[agent].tolist()
        
        # Record miscellaneous information.
        info_dict["reference_position"] = self.reference_state.tolist()
        info_dict["reference_velocity"] = self.alpha.tolist()

        # Check truncation conditions.
        if lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles):
            for agent in self.agents:
                info_dict[agent]["ready"] = True
                truncated_dict[agent] = True
            self.agents = []
        
        done_dict = {agent: self.dones[agent] for agent in self.possible_agents}

        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def _getAvailableActions(self, agent):
        ''' Returns the available actions for the given agent. '''

        return None