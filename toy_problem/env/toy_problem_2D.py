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

from toy_problem.env.entity import ENTITY_TYPE, Agent, Adversary


def add_args(parser):
    ''' Adds environment arguments. '''
    
    import argparse
    parser.add_argument("--num_adversaries", type=int, default=1,
                        help="The number of adversaries in the environment.")
    parser.add_argument("--random_start_positions", action=argparse.BooleanOptionalAction, default=False,
                        help="If true, agents will start at random positions in the world. If false, they will start at [0,0].")
    parser.add_argument("--state_per_agent", action=argparse.BooleanOptionalAction, default=False,
                        help="If true, the state function will return a separate copy of the state for each agent. "
                             "This is useful for centralized training, decentralized execution. "
                             "If false, the state function will return a single copy of the state that is shared among all agents.")
    parser.add_argument("--observation_probability", type=float, default=1.0,
                        help="The probability that an agent will observe another entity.")


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
    }


    def __init__(self,
                 num_agents = 1,
                 num_adversaries: int = 1,
                 max_cycles: int = -1,
                 world_size: float = 50.0,
                 random_start_positions: bool = False,
                 state_per_agent: bool = False,
                 observation_probability: float = 1.0,
                ):
        """
        Initialize the environment.
        """
        super().__init__()

        # Configuration.
        self.max_cycles = max_cycles
        self.world_dims = np.array([world_size, world_size], dtype=np.float32)
        self.random_start_positions = random_start_positions
        self.state_per_agent = state_per_agent
        self.observation_probability = observation_probability

        # Set up entities.
        self.possible_agents = [
            Agent(
                position=self.get_random_position(),
            ) for i in range(num_agents)
        ]
        self.possible_adversaries = [
            Adversary(
                position=self.get_random_position(),
            ) for i in range(num_adversaries)
        ]

        # Create the action space.
        action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore
        
        # Create the observation space.
        obs_space = {
            "adversaries": spaces.Dict({
                a: spaces.Dict({
                    "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                    "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                }) for a in self.possible_adversaries
            }),
            "agents": spaces.Dict({
                a: spaces.Dict({
                    "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                    "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                }) for a in self.possible_agents
            }),
            "id": spaces.Discrete(len(self.possible_agents)),
        }
        obs_space = spaces.Dict(obs_space)
        self.observation_spaces = spaces.Dict({agent: obs_space for agent in self.possible_agents}) # type: ignore

        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        if self.state_per_agent:
            self.state_space = self.observation_spaces
        else:
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
        for agent in self.agents:
            start_position = self.get_random_position() if self.random_start_positions else np.array([0.0, 0.0], dtype=np.float32)
            agent.reset(
                reset_start_position=True,
                position=start_position
            )
        
        # Reset the adversaries.
        self.adversaries = copy(self.possible_adversaries)
        for adversary in self.adversaries:
            start_position = self.get_random_position() if self.random_start_positions else np.array([0.0, 0.0], dtype=np.float32)
            adversary.reset(
                reset_start_position=True,
                position=start_position,
            )
            adversary.velocity = np.random.uniform(-1.0, 1.0, size=2)  # Random initial velocity
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)

        self.state_history = {
            a: [copy(a.position)] for a in self.agents + self.adversaries
        }

        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.agents}

        info = {
            agent: {
                "ready": True
            } for agent in self.agents
        }

        # Return the initial observation.
        observation = {}
        for agent in self.agents:
            obs, obs_mask = self.observe(agent)
            observation[agent] = obs
            info[agent]["visibility_mask"] = obs_mask

        return observation, info


    def get_random_position(self):
        ''' Returns a random position in the world. '''

        return np.random.uniform(-self.world_dims / 2, self.world_dims / 2)


    def render(self, pred=None, figsize=(9, 6), history_length=2):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches.
                
            Returns:
                None
        '''

        # Convert the predicted state back into a dictionary (unflatten).
        pred_unflattened = []
        pred_steps = pred.shape[0] if pred is not None else 0
        for i in range(pred_steps):
            p = spaces.unflatten(self.observation_spaces, pred[i].flatten())
            pred_unflattened.append(p)

        # print(f"Prediction: {pred}")

        # Plot as a line graph using matplotlib.
        plt.figure(figsize=figsize)

        # Set the axis limits.
        plt.xlim(-self.world_dims[0], self.world_dims[0])
        plt.ylim(-self.world_dims[1], self.world_dims[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.title("2D Leader-Follower Environment")
        plt.grid()
        
        # Plot the agent positions.
        positions = [a.position for a in self.possible_agents]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'bo', label='Followers')
        for i, agent in enumerate(self.agents):
            plt.annotate(f"{agent}", (positions[i][0] + 1, positions[i][1]), fontsize=8, color='blue')

            # Plot actual history for the agent.
            history = self.state_history[agent]
            plt.plot([h[0] for h in history], [h[1] for h in history], 'b', alpha=0.5, linewidth=0.5, label=f"{agent} actual")            
        
        # Plot the adversary positions.
        positions = [a.position for a in self.possible_adversaries]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'ro', label='Leaders')
        for i, adversary in enumerate(self.adversaries):
            plt.annotate(f"{adversary}", (positions[i][0] + 1, positions[i][1]), fontsize=8, color='red')        

            # Plot actual history for the adversary.
            history = self.state_history[adversary]
            plt.plot([h[0] for h in history], [h[1] for h in history], 'r', alpha=0.5, linewidth=0.5, label=f"{adversary} actual")


        # Plot history of predictions from the perspective of agent 0.
        if len(pred_unflattened) > 0:
            agent_preds = [pred_unflattened[i][self.possible_agents[0]]["agents"] for i in range(len(pred_unflattened))]
            for i, agent in enumerate(self.possible_agents):
                # Get the history of predictions for this agent.
                history = [p[agent]["position"] for p in agent_preds]
                plt.plot([h[0] for h in history], [h[1] for h in history], 'b--', alpha=0.5, linewidth=1.5)            
                plt.annotate(f"Pred {agent}", (history[-1][0] + 1, history[-1][1]), fontsize=8, color='blue')
            
            adversary_preds = [pred_unflattened[i][self.possible_agents[0]]["adversaries"] for i in range(len(pred_unflattened))]
            for i, adversary in enumerate(self.possible_adversaries):
                # Get the history of predictions for this adversary.
                history = [p[adversary]["position"] for p in adversary_preds]
                plt.plot([h[0] for h in history], [h[1] for h in history], 'r--', alpha=0.5, linewidth=1.5)            
                plt.annotate(f"Pred {adversary}", (history[-1][0] + 1, history[-1][1]), fontsize=8, color='red')

        # Add legend outside the plot.
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        plt.tight_layout()

        # Show the plot.
        plt.show()      


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


    def state(self):
        ''' Returns the global state of the environment.
            This is useful for centralized training, decentralized execution. '''
        
        if self.state_per_agent:
            # Return the state for each agent.
            return {a: self._populateStateSpace(a, force_visible=True)[0] for a in self.possible_agents}
        else:
            return self._populateStateSpace(self.possible_agents[0], force_visible=True)[0]


    def observe(self, agent, radius=None, allow_done_agents=False):
        ''' Returns the observation for the given agent.'''

        return self._populateStateSpace(agent)


    def available_actions(self, agent):
        ''' Returns the dictionary of available actions for all agents.
            This is not standard in the Pettingzoo API but is useful. '''
        return self.available_actions_dict[agent]


    def _populateStateSpace(self, agent, force_visible=False):
        ''' Returns a populated state/observation space.'''

        obs = {
            "adversaries": {
                a: {
                    "position": a.position,
                    "velocity": a.velocity,
                } for a in self.possible_adversaries
            },
            "agents": {
                a: {
                    "position": a.position,
                    "velocity": a.velocity,
                } for a in self.possible_agents
            },
            "id": self.possible_agents.index(agent),
        }
        
        # Create a visibility mask for the agents.
        def visible(entity):
            if force_visible:
                return True
            if entity == agent:
                return True
            if random.random() < self.observation_probability:
                return True
            return False
        obs_mask = {
            "adversaries": {},
            "agents": {},
            "id": True
        }
        for a in self.possible_adversaries:
            vis = visible(a)
            obs_mask["adversaries"][a] = {
                "position": vis,
                "velocity": vis,
            }
        for a in self.possible_agents:
            vis = visible(a)
            obs_mask["agents"][a] = {
                "position": vis,
                "velocity": vis,
            }

        return obs, obs_mask
    
    def _update_adversary_velocity(self, adversary):
        """
        Update velocity based on the step count to create a more complex pattern.
        This function creates a time-varying velocity that follows different patterns.
        """
        if random.random() < 0.3:
            # Randomly change velocity to create a new pattern
            adversary.velocity += np.random.normal(0, 0.3, size=2)
                
        # # Add some random noise to make the trajectory more natural
        noise_magnitude = 0.05 * min(1.0, self.step_count / 50.0)  # Gradually increase noise
        adversary.velocity += np.random.normal(0, noise_magnitude, size=2)

        # Normalize velocity to keep it within a reasonable range.
        norm = np.linalg.norm(adversary.velocity)
        if norm > 1.0:
            adversary.velocity = adversary.velocity / norm
                
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
                
        obs_dict = {}
        reward_dict = {agent: 0.0 for agent in self.possible_agents}
        truncated_dict = {agent: False for agent in self.possible_agents}
        info_dict = {
            agent: {
                "ready": True
            } for agent in self.possible_agents
        }

        # Perform agent actions.
        for agent in self.agents:
            if agent in action_dict:
                action = action_dict[agent]

                # Increment the agent state.                
                agent.position += action

                self.state_history[agent].append(agent.position.copy())

        # Perform adversary actions.
        for adversary in self.adversaries:
            self._update_adversary_velocity(adversary)
            adversary.position += adversary.velocity
            self.state_history[adversary].append(adversary.position.copy())

        # Provide penalty to all agents based on distance of each adversary to the closest agent.
        for adversary in self.adversaries:
            closest_agent = min(self.agents, key=lambda a: np.linalg.norm(adversary.position - a.position))
            distance = np.linalg.norm(adversary.position - closest_agent.position)
            # Reward is negative distance to encourage agents to stay close to adversaries.
            rwd = -distance
        for agent in self.agents:
            reward_dict[agent] += rwd

        # Perform observations.
        for agent in self.possible_agents:
            agent_observation, obs_mask = self.observe(agent)
            obs_dict[agent] = agent_observation
            info_dict[agent]["visibility_mask"] = obs_mask

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