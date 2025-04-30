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
                 world_size: float = 20.0,
                ):
        """
        Initialize the environment.
        """
        super().__init__()

        # Configuration.
        self.max_cycles = max_cycles
        self.world_dims = np.array([world_size, world_size], dtype=np.float32)
        num_adversaries = 1 #just 1 leader for now

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
        # obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + num_agents*2, ), dtype=np.float32)
        obs_space = spaces.Dict({
            "adversaries": spaces.Dict({
                adversary: spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32) for adversary in self.possible_adversaries
            }),
            "alpha": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "agents": spaces.Dict({
                agent: spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32) for agent in self.possible_agents
            }),
        })
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
        for agent in self.agents:
            agent.reset(
                reset_start_position=True,
                position=self.get_random_position()
            )
        
        # Reset the adversaries.
        self.adversaries = copy(self.possible_adversaries)
        for adversary in self.adversaries:
            adversary.reset(
                reset_start_position=True,
                position=self.get_random_position(),
            )
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)
        
        # Initialize alpha (velocity) and save base value for variations
        self.alpha = np.array([random.random(), random.random()], dtype=np.float32)
        self.base_alpha = self.alpha.copy()  # Store the initial alpha for reference
        self.alpha_history = [self.alpha.copy()]
        
        self.state_history = {
            a: [np.array([0.0, 0.0], dtype=np.float32)] for a in self.agents + self.adversaries
        }

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


    def get_random_position(self):
        ''' Returns a random position in the world. '''

        # Just return [0,0] for now. Compatible with the existing trained policy.
        # Remove when we are ready to train a new policy.
        return np.array([0.0, 0.0], dtype=np.float32)

        # return np.random.uniform(-self.world_dims / 2, self.world_dims / 2)


    def render(self, pred, figsize=(9, 6), history_length=2):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches.
                
            Returns:
                None
        '''


        # Convert the predicted state back into a dictionary (unflatten).
        pred_unflattened = []
        pred_steps = len(pred)
        for i in range(pred_steps):
            p = spaces.unflatten(self.observation_spaces, pred[i])
            pred_unflattened.append(p)

        # Plot as a line graph using matplotlib.
        plt.figure(figsize=figsize)

        # Set the axis limits.
        plt.xlim(-self.world_dims[0] / 2, self.world_dims[0] / 2)
        plt.ylim(-self.world_dims[1] / 2, self.world_dims[1] / 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.title("2D Leader-Follower Environment")
        plt.grid()
        
        # Plot the goal positions.
        # for goal in self.goals:
        #     if goal.state == GoalZone.GOAL_STATE.UNREACHED:
        #         color = 'grey'
        #     elif goal.state == GoalZone.GOAL_STATE.REACHED_AGENT:
        #         color = 'green'
        #     elif goal.state == GoalZone.GOAL_STATE.REACHED_ADVERSARY:
        #         color = 'red'
        #     label = f"Goal {goal.entity_id}"
        #     marker = plt.Circle(goal.position, goal.radius, color=color, alpha=0.5, label=label)
        #     plt.gca().add_artist(marker)
        # positions = [g.position for g in self.goals]
        # plt.plot([p[0] for p in positions], [p[1] for p in positions], 'go', label='Goals', markersize=self.goals[0].radius*10)

        # Plot the agent positions.
        positions = [a.position for a in self.possible_agents]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'bo', label='Followers')
        for i, agent in enumerate(self.agents):
            plt.annotate(f"{agent}", (positions[i][0] + 1, positions[i][1]), fontsize=8, color='blue')

            # Plot actual history for the agent.
            history = self.state_history[agent]
            plt.plot([h[0] for h in history], [h[1] for h in history], 'b', alpha=0.5, linewidth=0.5)            
        
        # Plot the adversary positions.
        positions = [a.position for a in self.possible_adversaries]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'ro', label='Leaders')
        for i, adversary in enumerate(self.adversaries):
            plt.annotate(f"{adversary}", (positions[i][0] + 1, positions[i][1]), fontsize=8, color='red')        

            # Plot actual history for the adversary.
            history = self.state_history[adversary]
            plt.plot([h[0] for h in history], [h[1] for h in history], 'r', alpha=0.5, linewidth=0.5)


        # Plot history of predictions from the perspective of agent 0.
        if len(pred_unflattened) > 0:
            agent_preds = [pred_unflattened[i][self.possible_agents[0]]["agents"] for i in range(len(pred_unflattened))]
            for i, agent in enumerate(self.possible_agents):
                # Get the history of predictions for this agent.
                history = [p[agent] for p in agent_preds]
                plt.plot([h[0] for h in history], [h[1] for h in history], 'b--', alpha=0.5, linewidth=1.5)            
                plt.annotate(f"Pred {agent}", (history[-1][0] + 1, history[-1][1]), fontsize=8, color='blue')
            
            adversary_preds = [pred_unflattened[i][self.possible_agents[0]]["adversaries"] for i in range(len(pred_unflattened))]
            for i, adversary in enumerate(self.possible_adversaries):
                # Get the history of predictions for this adversary.
                history = [p[adversary] for p in adversary_preds]
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

        # obs = np.array([a.position for a in self.possible_adversaries] + [self.alpha] + [a.position for a in self.possible_agents], dtype=np.float32)
        # obs = obs.flatten()

        obs = {
            "adversaries": {a: a.position for a in self.possible_adversaries},
            "alpha": self.alpha,
            "agents": {a: a.position for a in self.possible_agents},
        }

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
                agent.position += action

                self.state_history[agent].append(agent.position.copy())

        # Increment the reference state using the time-varying alpha
        for adversary in self.adversaries:
            adversary.position += self.alpha
            self.state_history[adversary].append(adversary.position.copy())

        # Provide reward based on distance to reference state
        meanAdversary = np.mean([a.position for a in self.adversaries], axis=0)
        for agent in self.agents:
            distance = np.linalg.norm(agent.position - meanAdversary)
            reward_dict[agent] = -distance

        # Perform observations.
        for agent in self.possible_agents:
            agent_observation = self.observe(agent)
            obs_dict[agent] = agent_observation
            info_dict[f"distance/{agent}"] = np.linalg.norm(agent.position - meanAdversary)
            info_dict[f"position/{agent}"] = agent.position.tolist()
        
        # Record miscellaneous information.
        info_dict["reference_position"] = meanAdversary.tolist()
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