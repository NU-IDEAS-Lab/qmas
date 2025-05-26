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
                 num_adversaries = 3,
                 num_goals = 6,
                #  max_cycles: int = -1,
                 render_mode: str = "human",
                 world_size: float = 100.0,
                 partially_observable: bool = False
                ):
        """
        Initialize the environment.
        """
        super().__init__()

        # Configuration.
        # self.max_cycles = max_cycles
        self.world_dims = np.array([world_size, world_size])
        self.render_mode = render_mode
        self.partially_observable = partially_observable

        # Create the agents.
        self.possible_agents = [
            Agent(
                position=self.get_random_position(),
                speed_max=2.0,
            ) for i in range(num_agents)
        ]
        self.possible_adversaries = [
            Adversary(
                position=self.get_random_position(),
                speed_max=1.0,
            ) for i in range(num_adversaries)
        ]
        self.possible_goals = [
            GoalZone(
                position=self.get_random_position(),
                radius=2.0
            ) for i in range(num_goals)
        ]

        space_r2 = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Create the action space.
        action_space = space_r2
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore
        
        # Create the observation space.
        obs_space_dict = {
            "id": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.int32
            ),
            "adversary_states": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.possible_adversaries), 2 + 2), # position + target
                dtype=np.float32
            ),
            "agent_states": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.possible_agents), 2),
                dtype=np.float32
            ),
            "goal_states": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.possible_goals), 2 + 1), # position + state
                dtype=np.float32
            )
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

        # Reset the goals.
        self.goals = copy(self.possible_goals)
        for goal in self.goals:
            goal.reset(
                reset_start_position=True,
                position=self.get_random_position()
            )

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        for agent in self.agents:
            agent.reset(
                reset_start_position=True,
                position=self.get_random_position()
            )
        
        # Reset the adversaries.
        self.adversaries = copy(self.possible_adversaries)
        targets = random.sample(self.goals, len(self.adversaries))
        for adversary in self.adversaries:
            adversary.reset(
                reset_start_position=True,
                position=self.get_random_position(),
                target=targets.pop(0)
            )
        
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


    def get_random_position(self):
        ''' Returns a random position in the world. '''

        return np.random.uniform(-self.world_dims / 2, self.world_dims / 2)

    def render(self, figsize=(9, 6)):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches.
                
            Returns:
                None
        '''

        # Plot as a line graph using matplotlib.
        plt.figure(figsize=figsize)

        # Set the axis limits.
        plt.xlim(-self.world_dims[0] / 2, self.world_dims[0] / 2)
        plt.ylim(-self.world_dims[1] / 2, self.world_dims[1] / 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.title("Simpler Adversary Environment")
        plt.grid()
        
        # Plot the goal positions.
        for goal in self.goals:
            if goal.state == GoalZone.GOAL_STATE.UNREACHED:
                color = 'grey'
            elif goal.state == GoalZone.GOAL_STATE.REACHED_AGENT:
                color = 'green'
            elif goal.state == GoalZone.GOAL_STATE.REACHED_ADVERSARY:
                color = 'red'
            label = f"Goal {goal.entity_id}"
            marker = plt.Circle(goal.position, goal.radius, color=color, alpha=0.5, label=label)
            plt.gca().add_artist(marker)
        # positions = [g.position for g in self.goals]
        # plt.plot([p[0] for p in positions], [p[1] for p in positions], 'go', label='Goals', markersize=self.goals[0].radius*10)

        # Plot the agent positions.
        positions = [a.position for a in self.possible_agents]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'bo', label='Agents')
        for i, agent in enumerate(self.agents):
            plt.annotate(f"{agent}", (positions[i][0] + 1, positions[i][1]), fontsize=8, color='blue')
        
        # Plot the adversary positions.
        positions = [a.position for a in self.possible_adversaries]
        plt.plot([p[0] for p in positions], [p[1] for p in positions], 'ro', label='Adversaries')
        for i, adversary in enumerate(self.adversaries):
            plt.annotate(f"{adversary}", (positions[i][0] + 1, positions[i][1]), fontsize=8, color='red')        

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


    # def available_actions_space(self, agent):
    #     ''' Generate a Space for the available actions, given the action space. '''

    #     action_space = self.action_space(agent)
    #     def get_available_action_space(action_space):
    #         if action_space.__class__.__name__ in ["Tuple", "Dict"]:
    #             return spaces.Dict({k: get_available_action_space(v) for k, v in action_space.spaces.items()})
    #         elif action_space.__class__.__name__ == "Discrete":
    #             return spaces.MultiBinary(action_space.n)
    #         elif action_space.__class__.__name__ == "MultiDiscrete":
    #             return spaces.MultiBinary(len(action_space.nvec), np.max(action_space.nvec))
    #         else:
    #             raise NotImplementedError(f"Action space {action_space} not supported for action masking.")
    #     return get_available_action_space(action_space)


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


    # def available_actions(self, agent):
    #     ''' Returns the dictionary of available actions for all agents.
    #         This is not standard in the Pettingzoo API but is useful. '''
    #     return self.available_actions_dict[agent]


    def _populateStateSpace(self, agent):
        ''' Returns a populated state/observation space.'''

        obs = {}

        def normalize_position(position):
            ''' Normalizes the position to be between -1 and 1. '''
            
            # Perform min-max normalization.
            norm_position = (position - (-self.world_dims / 2)) / (self.world_dims)
            norm_position = norm_position * 2 - 1
            return norm_position

        # ID
        obs["id"] = np.array([agent.entity_id], dtype=np.int32)

        # Adversary states.
        obs["adversary_states"] = np.zeros((len(self.possible_adversaries), 2 + 2), dtype=np.float32)
        for i, adversary in enumerate(self.possible_adversaries):
            obs["adversary_states"][i, 0:2] = normalize_position(adversary.position)
            if self.partially_observable:
                obs["adversary_states"][i, 2:4] = np.ones((2,), dtype=np.float32) * self.metadata["null_value"]
            else:
                obs["adversary_states"][i, 2:4] = normalize_position(adversary.target.position)

        # Agent states. Ensure current agent is always first.
        obs["agent_states"] = np.zeros((len(self.possible_agents), 2), dtype=np.float32)
        obs["agent_states"][0, :] = normalize_position(agent.position)
        idx = 1
        for other_agent in self.possible_agents:
            if other_agent != agent:
                obs["agent_states"][idx, :] = normalize_position(other_agent.position)
                idx += 1

        # Goal states.
        obs["goal_states"] = np.zeros((len(self.possible_goals), 2 + 1), dtype=np.float32)
        for i, goal in enumerate(self.possible_goals):
            obs["goal_states"][i, 0:2] = normalize_position(goal.position)
            obs["goal_states"][i, 2] = goal.state.value

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

                if np.linalg.norm(action) > 1.0:
                    unit_action = action / np.linalg.norm(action)
                    action = unit_action * agent.speed_max

                # Check if the action is valid.
                # if not self.action_space(agent).contains([action]):
                #     raise ValueError(f"Invalid action {action} of type {type(action)} provided.")

                # Update the agent's state.
                agent.velocity = action
                agent.position += agent.velocity

                # Check whether agent reached any goals.
                for goal in self.goals:
                    goal.reached_check(agent)
                    # reached = goal.reached_check(agent)
                    # if reached:
                    #     # Provide reward.
                    #     reward_dict[agent] += 1.0


        # Update the adversaries.
        self.step_adversaries()

        # Provide reward.
        for agent in self.agents:
            reward_dict[agent] = self.reward(agent)

        # Perform observations.
        for agent in self.possible_agents:
            obs_dict[agent] = self.observe(agent)
        
        # Record miscellaneous information.
        info_dict["agent_count"] = len(self.agents)
        # for agent in self.agents:
        #     info_dict[f"x/{agent}"] = self.agent_states[agent]
        # info_dict["x/reference"] = self.reference_state

        # Check whether all targets are reached.
        done = True
        for adversary in self.adversaries:
            goal = adversary.target
            if goal.state == GoalZone.GOAL_STATE.UNREACHED:
                done = False
                break
        
        # End the game.
        if done:
            for agent in self.agents:
                self.dones[agent] = True
            self.agents = []
        done_dict = {agent: self.dones[agent] for agent in self.possible_agents}

        # Set available actions.
        # self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def step_adversaries(self):
        ''' Updates the adversaries. '''

        for adversary in self.adversaries:
            target = adversary.target
            if target is not None:
                # Move towards the target.
                direction = target.position - adversary.position
                direction /= np.linalg.norm(direction)
                adversary.velocity = direction * adversary.speed_max
                adversary.position += adversary.velocity

            # Check whether any goals have been reached.
            for goal in self.goals:
                goal.reached_check(adversary)


    def reward(self, agent):
        ''' Returns the reward for the given agent. '''

        rwd = 0.0

        WEIGHT_ADVERSARY_DIST_PENALTY = 1.0
        WEIGHT_AGENT_DIST_REWARD = 0.1

        pos_prev = agent.position - agent.velocity
        pos_curr = agent.position

        # # Reward for moving towards adversary targets.
        # for adversary in self.adversaries:
        #     target = adversary.target
        #     if target.state == GoalZone.GOAL_STATE.UNREACHED:
        #         dist_prev = np.linalg.norm(target.position - pos_prev)
        #         dist_curr = np.linalg.norm(target.position - pos_curr)
        #         rwd += WEIGHT_AGENT_DIST_REWARD * max(dist_prev - dist_curr, 0.0) / agent.speed_max

        # # Provide penalty for adversary moving towards an unvisited target.
        # for adversary in self.adversaries:
        #     target = adversary.target
        #     if target.state == GoalZone.GOAL_STATE.UNREACHED:
        #         adversary_dist_prev = np.linalg.norm(target.position - adversary.position - adversary.velocity)
        #         adversary_dist_curr = np.linalg.norm(target.position - adversary.position)
        #         rwd += WEIGHT_ADVERSARY_DIST_PENALTY * min(adversary_dist_prev - adversary_dist_curr, 0.0) / adversary.speed_max
        

        # Provide reward based on shorter distance to (20.0, 20.0)
        dist = np.linalg.norm(agent.position - np.array([20.0, 20.0]))
        rwd += 1.0 / (dist + 1e-6) # Avoid division by zero

        return rwd


    # def _getAvailableActions(self, agent):
    #     ''' Returns the available actions for the given agent. '''

    #     return None