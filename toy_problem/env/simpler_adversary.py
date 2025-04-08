from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from gymnasium import spaces
import random
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from copy import copy
from enum import IntEnum

from toy_problem.env.entity import Agent, Adversary, GoalZone


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
        "name": "simpler_adversary_v0",
        "render_modes": ["human", "rgb_array"],
        "null_value": -1.0
    }


    def __init__(self,
                 num_agents = 3,
                 num_adversaries = 5,
                 num_goals = 5,
                 max_cycles: int = -1,
                ):
        """
        Initialize the environment.
        """
        super().__init__()

        # Configuration.
        self.max_cycles = max_cycles

        # Create the agents.
        self.possible_agents = [Agent() for i in range(num_agents)]
        self.possible_adversaries = [Adversary() for i in range(num_adversaries)]
        self.possible_goals = [GoalZone() for i in range(num_goals)]

        space_r2 = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Create the action space.
        action_space = space_r2
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore
        
        # Create the observation space.
        obs_space_dict = {
            "adversary_states": spaces.Dict({
                b: space_r2 for b in self.possible_adversaries
            }),
            "agent_states": spaces.Dict({
                a: space_r2 for a in self.possible_agents
            }),
            "goal_states": spaces.Dict({
                g: space_r2 for g in self.possible_goals
            }),
        }
        obs_space_dict_sorted = {k: obs_space_dict[k] for k in sorted(obs_space_dict.keys())}
        obs_space = spaces.Dict(obs_space_dict_sorted)
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
            agent.reset()
        
        # Reset the adversaries.
        self.adversaries = copy(self.possible_adversaries)
        for adversary in self.adversaries:
            adversary.reset()
        
        # Reset the goals.
        self.goals = copy(self.possible_goals)
        for goal in self.goals:
            goal.reset()

        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)

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
        
        # Plot the agent positions.
        positions = [a.position for a in self.agents]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'bo', label='Agents')
        for i, agent in enumerate(self.agents):
            plt.annotate(f"Agent {i}", (positions[i][0], positions[i][1]), fontsize=8, color='blue')
        
        # Plot the adversary positions.
        positions = [a.position for a in self.adversaries]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'ro', label='Adversaries')
        for i, adversary in enumerate(self.adversaries):
            plt.annotate(f"Adversary {i}", (positions[i][0], positions[i][1]), fontsize=8, color='red')
        
        # Plot the goal positions.
        positions = [g.position for g in self.goals]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'go', label='Goals')

        # Add legend.
        plt.legend()            


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

        obs = {}

        # Adversary states.
        obs["adversary_states"] = {a: a.position for a in self.adversaries}

        # Agent states.
        obs["agent_states"] = {a: a.position for a in self.agents}

        # Goal states.
        obs["goal_states"] = {g: g.position for g in self.goals}

        # Ensure the order of the keys is consistent.
        obs_sorted = {k: obs[k] for k in sorted(obs.keys())}

        return obs_sorted
    

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

        # Perform actions.
        for agent in self.agents:
            if agent in action_dict:
                action = action_dict[agent]

                # Check if the action is valid.
                # if not self.action_space(agent).contains([action]):
                #     raise ValueError(f"Invalid action {action} of type {type(action)} provided.")

                # Increment the agent state.
                self.agent_states[agent] += action

                self.state_history[agent].append(self.agent_states[agent])

        # Increment the reference state.
        self.reference_state += self.alpha
        self.reference_state_history.append(self.reference_state)

        # Provide reward.
        for agent in self.agents:
            # reward_dict[agent] = -abs(action_dict[agent] - self.alpha)
            reward_dict[agent] = -abs(self.agent_states[agent] - self.reference_state)
            # reward_dict[agent] = -np.log(abs(self.agent_states[agent] - self.reference_state))

        # Perform observations.
        for agent in self.possible_agents:
            agent_observation = self.observe(agent)
            obs_dict[agent] = agent_observation
        
        # Record miscellaneous information.
        info_dict["agent_count"] = len(self.agents)
        info_dict["avg_distance"] = np.mean([abs(self.agent_states[agent] - self.reference_state) for agent in self.agents])
        # for agent in self.agents:
        #     info_dict[f"x/{agent}"] = self.agent_states[agent]
        # info_dict["x/reference"] = self.reference_state

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