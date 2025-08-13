from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import functools
from gymnasium import spaces
import random
import numpy as np
from matplotlib import pyplot as plt
from copy import copy

from isru_zoo.env.entity import ENTITY_TYPE, Agent, Depot
from isru_zoo.env.resource import TestResource1, TestResource2


def add_args(parser):
    ''' Adds environment arguments. '''
    
    import argparse
    parser.add_argument("--num_obstacles", type=int, default=20,
                        help="The number of obstacles to place in the world.")
    parser.add_argument("--world_size", type=int, default=50,
                        help="The size of the world. The world is a square with side length `world_size`.")
    parser.add_argument("--render_mode", type=str, default="human",
                        choices=parallel_env.metadata["render_modes"],
                        help="The rendering mode for the environment.")
    


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
        "name": "isru_v0",
        "render_modes": ["human", "rgb_array"],
    }


    def __init__(self,
            num_agents = 1,
            max_cycles: int = -1,
            world_size: int = 50,
            num_obstacles: int = 10,
            render_mode: str = "human",
        ):
        """
        Initialize the environment.
        """
        super().__init__()

        # Configuration.
        self.max_cycles = max_cycles
        self.world_dims = np.array([world_size, world_size], dtype=np.int32)
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode

        # Set up entities.
        self.possible_agents = [
            Agent(
                position=self.get_random_position(),
            ) for i in range(num_agents)
        ]

        # Set up the possible resources.
        self.possible_resources = [
            TestResource1(10),
            TestResource2(5)
        ]

        # Set up depots.
        self.possible_depots = [
            Depot(
                position=self.get_random_position(),
                resource = r
            ) for r in self.possible_resources
        ]

        # Set up spaces.
        self.observation_spaces = spaces.Dict({
            agent: self.observation_space(agent) for agent in self.possible_agents
        })
        self.action_spaces = spaces.Dict({
            agent: self.action_space(agent) for agent in self.possible_agents
        })
        
        # Reset the environment.
        self.reset()


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        if seed != None:
            random.seed(seed)
        
        origin = np.array([0.0, 0.0], dtype=np.float32)
        world_indices_x = np.arange(self.world_dims[0])
        world_indices_y = np.arange(self.world_dims[1])

        # Reset obstacles.
        self.map_obstacles = np.zeros(self.world_dims, dtype=np.float32)
        self.map_obstacles[
            np.random.choice(world_indices_x, self.num_obstacles),
            np.random.choice(world_indices_y, self.num_obstacles)
        ] = 1.0
        positions_available = np.argwhere(self.map_obstacles == 0)

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        for agent in self.agents:
            idx = np.random.randint(positions_available.shape[0])
            start_position = positions_available[idx]
            agent.reset(
                reset_start_position=True,
                position=start_position
            )

        # Reset depots.
        # Ensure they are placed in an available location.
        self.map_depots = np.zeros(self.world_dims, dtype=np.float32)
        for depot in self.possible_depots:
            idx = np.random.randint(positions_available.shape[0])
            depot.position = positions_available[idx]
            self.map_depots[depot.position[0], depot.position[1]] = depot.resource_id
        
        # Reset resources.
        self.map_resources = {}
        for r in self.possible_resources:
            self.map_resources[r] = np.zeros(self.world_dims, dtype=np.float32)
            for _ in range(r.quantity):
                idx = np.random.randint(positions_available.shape[0])
                pos = positions_available[idx]
                self.map_resources[r][pos] += 1.0

        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)

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

        return np.random.uniform(-self.world_dims / 2, self.world_dims / 2).astype(np.float32)


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

        # Get the true environment state.
        state = self.state()

        # Plot state as a grid using matplotlib.
        plt.figure(figsize=figsize)

        # Plot the map layers.
        map_layers = state["map"]
        num_layers = map_layers.shape[-1]
        for i in range(num_layers):
            plt.subplot(1, num_layers, i + 1)
            plt.imshow(map_layers[:, :, i], cmap="gray", origin="lower")
            plt.title(f"Layer {i}")
            plt.axis("off")
        
        # Display the total reward for this step. Position this text below the subplots. Do not use suptitle.
        plt.figtext(0.5, 0.01, f"Step: {self.step_count}, Total Reward: {sum([self.get_reward(agent, False, False) for agent in self.agents]):.2f}", ha="center", fontsize=8)

        if self.render_mode == "human":
            # Show the plot.
            plt.show()
            return None
        elif self.render_mode == "rgb_array":
            # Save the plot to a buffer and return it as an RGB array.
            from io import BytesIO
            io_buf = BytesIO()
            plt.savefig(io_buf, format='raw')
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(plt.bbox.bounds[3]), int(plt.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return img_arr

    @property
    @functools.cache
    def state_space(self):
        ''' Returns the state space of the environment. '''

        # Determine number of layers in the map.
        # Layers are: obstacles, agents, depots, resources.
        num_layers = 1 + 1 + 1 + len(self.possible_resources)

        # Create the state space.
        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        return spaces.Dict({
            "id": spaces.Discrete(len(self.possible_agents)),
            "map": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(*self.world_dims, num_layers),
                dtype=np.float32
            )
        })


    @functools.cache
    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''

        return self.state_space


    @functools.cache
    def action_space(self, agent):
        ''' Returns the action space for the given agent. '''
        
        return spaces.Dict({
            # Movement is specified by relative motion in two dimensions.
            # The agent can only move one space at a time.
            "movement": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),

            # Resource actions are represented as a floating point value for each resource type.
            # To pick up resources, the agent uses a positive number.
            # To drop resources, the agent uses a negative number.
            "resources": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.possible_resources),), dtype=np.float32),
        })


    @functools.cache
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

        # Load agent data into a map.
        map_agents = np.zeros(self.world_dims, dtype=np.int32)
        for i, a in enumerate(self.agents):
            pos = a.position.astype(np.int32)
            map_agents[pos[0], pos[1]] = i + 1  # Start from 1 to avoid confusion with empty space.

        # Build the combined map.
        layers = [self.map_obstacles, map_agents, self.map_depots, *self.map_resources.values()]
        map_combined = np.stack(layers, axis=-1)

        # Create the observation.
        obs = {
            "id": self.possible_agents.index(agent),
            "map": map_combined
        }

        obs_mask = {
            "id": True,
            "map": np.ones_like(map_combined, dtype=bool)
        }
        
        return obs, obs_mask
    
    def _update_adversary_velocity(self, adversary):
        """
        Update velocity based on the step count to create a more complex pattern.
        This function creates a time-varying velocity that follows different patterns.
        """
        if random.random() < 0.3:
            # Randomly change velocity to create a new pattern
            adversary.velocity += np.random.normal(0, 0.3, size=self.num_dimensions)
                
        # # Add some random noise to make the trajectory more natural
        noise_magnitude = 0.05 * min(1.0, self.step_count / 50.0)  # Gradually increase noise
        adversary.velocity += np.random.normal(0, noise_magnitude, size=self.num_dimensions)

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
                # Parse the action.
                action = spaces.unflatten(self.action_space(agent), action_dict[agent])
                agent.last_action = action

                # Set agent velocity.
                agent.velocity = action["movement"].astype(np.int32)
                agent.velocity = np.clip(agent.velocity, -1.0, 1.0)

                # Move the agent.
                raw_position=agent.position + agent.velocity
                pos_min = np.array([0.0, 0.0], dtype=np.float32)
                pos_max = self.world_dims.astype(np.float32) - 1.0
                new_position = np.clip(raw_position, pos_min, pos_max)
                new_position_int = new_position.astype(np.int32)
                if self.map_obstacles[new_position_int[0], new_position_int[1]] == 0:
                    agent.position = new_position
                # Handle the resource actions.
                # TODO: Implement resource pickup/dropoff using the action["resources"] values.
                # for r in self.map_resources.keys():
                #     if self.map_resources[r][agent.position[0], agent.position[1]]==1 and isinstance(agent, Extractor):
                #         if action['resources']==0:
                #             agent.resources[r]+=1
                #             self.map_resources[r][agent.position[0], agent.position[1]]=0
                
                # for depot in self.possible_depots: 
                #     if self.map_depots[agent.position[0], agent.position[1]]==depot.resource_id and isinstance(agent, Hauler):
                #         if action['resources']==1:
                #             depot.resources[depot.resource_id]=agent.resources[depot.resource_id]
                #             agent.resources[depot.resource_id]=0


        # Check termination conditions.
        end_truncate = lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles)
        end_done = False  # No done conditions for now.

        # Perform post-step calculations.
        for agent in self.possible_agents:
            agent_observation, obs_mask = self.observe(agent)
            obs_dict[agent] = agent_observation
            info_dict[agent]["visibility_mask"] = obs_mask

            # Get the reward for the agent.
            reward_dict[agent] = self.get_reward(agent, end_truncate, end_done)

        # Handle end of episode.
        if end_truncate or end_done:
            for agent in self.agents:
                info_dict[agent]["ready"] = True
                truncated_dict[agent] = True
            self.agents = []
        
        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, self.dones, truncated_dict, info_dict


    def get_reward(self, agent, end_truncate, end_done):
        ''' Returns the reward for the given agent. '''

        # Provide reward for the agent to be in proximity of (30, 30).
        target_position = np.array([2, 2], dtype=np.float32)
        distance = np.linalg.norm(agent.position - target_position)
        reward = -distance

        # Provide a bonus reward if the resource actions are both 1.
        # if agent.last_action is not None:
        #     resource_actions = agent.last_action["resources"]
        #     if np.allclose(resource_actions, 1.0, atol=0.1):
        #         reward += 1.0

        return reward


    def _getAvailableActions(self, agent):
        ''' Returns the available actions for the given agent. '''

        return None