from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import functools
from gymnasium import spaces
import random
import numpy as np
from matplotlib import pyplot as plt
from copy import copy

from isru_zoo.env.entity import ENTITY_TYPE, Agent, Depot, Extractor, Hauler, Prospector
from isru_zoo.env.resource import TestResource1, TestResource2


def add_args(parser):
    ''' Adds environment arguments. '''
    
    import argparse
    parser.add_argument("--num_extractors", type=int, default=2,
                        help="The number of extractor vehicles to place in the world.")
    parser.add_argument("--num_haulers", type=int, default=2,
                        help="The number of hauler vehicles to place in the world.")
    parser.add_argument("--num_prospectors", type=int, default=1,
                        help="The number of prospector vehicles to place in the world.")
    parser.add_argument("--num_obstacles", type=int, default=0,
                        help="The number of obstacles to place in the world.")
    parser.add_argument("--world_size", type=int, default=50,
                        help="The size of the world. The world is a square with side length `world_size`.")
    parser.add_argument("--observation_radius", type=int, default=10,
                        help="The radius within which agents can observe each other and resources.")
    parser.add_argument("--hauler_capacity", type=float, default=10.0,
                        help="The maximum amount of resources a hauler can carry.")
    parser.add_argument("--render_mode", type=str, default="human",
                        choices=parallel_env.metadata["render_modes"],
                        help="The rendering mode for the environment.")
    


def validate_args(parsed_args):
    ''' Validates the arguments. '''
    
    # Set the number of agents based on the number of each type.
    parsed_args.num_agents = \
        parsed_args.num_extractors + \
        parsed_args.num_haulers + \
        parsed_args.num_prospectors


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
            num_extractors: int = 2,
            num_haulers: int = 2,
            num_prospectors: int = 1,
            max_cycles: int = -1,
            num_obstacles: int = 10,
            world_size: int = 50,
            observation_radius: int = 10,
            hauler_capacity: float = 10.0,
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
        self.default_observation_radius = observation_radius
        self.default_hauler_capacity = hauler_capacity

        # Set up entities.
        self.possible_agents = \
            [Extractor(
                position=self.get_random_position(),
                observation_radius=self.default_observation_radius
            ) for _ in range(num_extractors)] + \
            [Hauler(
                position=self.get_random_position(),
                carry_capacity=self.default_hauler_capacity,
                observation_radius=self.default_observation_radius
            ) for _ in range(num_haulers)] + \
            [Prospector(
                position=self.get_random_position(),
                observation_radius=self.default_observation_radius
            ) for _ in range(num_prospectors)]

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

        # Record the map shape.
        self.map_shape = (*self.world_dims, len(self.possible_resources) + 3)  # +3 for obstacles, agents, depots

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
            agent.reward = 0.0

            # Haulers start empty: cargo per resource_id -> 0.0
            if isinstance(agent, Hauler):
                agent.cargo = {r.resource_id: 0.0 for r in self.possible_resources}

        # Reset depots.
        # Ensure they are placed in an available location.
        self.map_depots = np.zeros(self.world_dims, dtype=np.float32)
        for depot in self.possible_depots:
            idx = np.random.randint(positions_available.shape[0])
            depot.position = positions_available[idx]
            # reset per-episode depot accounting
            if not hasattr(depot, "stock"):
                depot.stock = 0.0
            else:
                depot.stock = 0.0
            self.map_depots[depot.position[0], depot.position[1]] = depot.resource_id
        
        # Build stable resource index mappings for action vector <-> resource objects
        self.resource_list = list(self.possible_resources)
        self.rid_to_idx = {r.resource_id: i for i, r in enumerate(self.resource_list)}
        self.idx_to_res = {i: r for i, r in enumerate(self.resource_list)}

        # Reset resources.
        self.map_resources = {}
        self.mask_map_resources_discovered = np.zeros(self.world_dims, dtype=bool)
        for r in self.possible_resources:
            self.map_resources[r] = np.zeros(self.world_dims, dtype=np.float32)
            for _ in range(r.quantity):
                idx = np.random.randint(positions_available.shape[0])
                pos = positions_available[idx]
                self.map_resources[r][pos[0], pos[1]] += 1.0

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
        # Get the observation with respect to agent 0.
        # state = self.observe(self.agents[0], senders=set())[0]

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

        # Create the state space.
        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        return spaces.Dict({
            "id": spaces.Discrete(len(self.possible_agents)),
            "map": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.map_shape,
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
            "movement": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),

            # Communication is a simple boolean flag.
            "communication": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),

            # Resource actions are represented as a floating point value for each resource type.
            # To pick up resources, the agent uses a positive number.
            # To drop resources, the agent uses a negative number.
            "resources": spaces.Box(low=-self.default_hauler_capacity, high=self.default_hauler_capacity, shape=(len(self.possible_resources),), dtype=np.int32),
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


    def observe(self, agent, senders=set()):
        ''' Returns the observation for the given agent.'''

        # Collect local data.
        local_obs, local_obs_mask = self._populateStateSpace(agent)

        # Set up the matrices.
        map = np.copy(local_obs["map"])
        map_mask = np.zeros_like(local_obs_mask["map"], dtype=bool)

        # Handle communicated data.
        for sender in senders:
            sender_obs, sender_obs_mask = self._populateStateSpace(sender)
            sender_visible = sender_obs_mask["map"] == True
            map[sender_visible] = sender_obs["map"][sender_visible]
            map_mask[sender_visible] = True
        
        # Apply local observations (overwrite any communicated data).
        map[local_obs_mask["map"] == True] = local_obs["map"][local_obs_mask["map"] == True]
        map_mask |= local_obs_mask["map"]

        # Update the local observation.
        combined_obs = local_obs
        combined_obs["map"] = map
        combined_obs_mask = local_obs_mask
        combined_obs_mask["map"] = map_mask

        # Debugging: highlight the visible area in the map.
        # obs["map"][obs_mask["map"], :] += 0.2

        return combined_obs, combined_obs_mask


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
            if a == agent and not force_visible:
                # Distinguish the agent in the map.
                map_agents[pos[0], pos[1]] = 255
            else:
                # Index from 1 to avoid confusion with empty space.
                map_agents[pos[0], pos[1]] = i + 1

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

        # Calculate the visible area based on a circular observation radius.
        if not force_visible or True:
            radius = agent.observation_radius
            pos = agent.position.astype(np.int32)
            visible = (np.arange(self.world_dims[0])[:, None] - pos[0]) ** 2 + \
                (np.arange(self.world_dims[1])[None, :] - pos[1]) ** 2 <= radius ** 2
            # obs["map"][visible, 0] += 0.2 # testing - show the visible area in the obstacle layer
            obs_mask["map"][~visible, :] = False

        # Vehicle-class visibility: only Prospectors can directly observe resources.
        # Layers are ordered as: 0=obstacles, 1=agents, 2=depots, 3+=resources
        resource_layer = 3
        if not force_visible:
            if not isinstance(agent, Prospector):
                # Mask out all resource layers for non-prospector local observations.
                # (Communicated observations will still be merged in `observe()`)
                obs_mask["map"][:, :, resource_layer:] = False

        # Update the discovered resources mask.
        if isinstance(agent, Prospector) and not force_visible:
            for i, r in enumerate(self.possible_resources):
                self.mask_map_resources_discovered |= (self.map_resources[r] > 0) & obs_mask["map"][:, :, resource_layer + i]
        
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
                "ready": True,
            } for agent in self.possible_agents
        }
        resources_deposited = {
            agent: {
                r: 0.0 for r in self.possible_resources
            } for agent in self.possible_agents
        }
        senders = set()

        # Perform agent actions.
        for agent in self.agents:
            if agent in action_dict:
                # Parse the action.
                action = spaces.unflatten(self.action_space(agent), action_dict[agent])

                # Check for action validity.
                if not self.action_space(agent).contains(action):
                    raise ValueError(f"Invalid action for agent {agent}: {action}")

                # Record the action.
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

                # Handle communication.
                if action["communication"][0] >= 0.5:
                    senders.add(agent)

                # Corrected resource handling for Hauler agents
                if isinstance(agent, Hauler):
                    action_vec = action["resources"]
                    px, py = agent.position.astype(np.int32)

                    cap = agent.capabilities.get("carry_capacity", 0.0)
                    current_load = sum(agent.cargo.values())
                    free = max(0.0, cap - current_load)

                    for idx, val in enumerate(action_vec):
                        if val > 0:
                            r = self.idx_to_res[idx]
                            available = self.map_resources[r][px, py]
                            want = float(val)
                            take = min(want, available, free)
                            if take > 0:
                                self.map_resources[r][px, py] -= take
                                agent.cargo[r.resource_id] = agent.cargo.get(r.resource_id, 0.0) + take
                                free -= take
                        elif val < 0:
                            r = self.idx_to_res[idx]
                            want_drop = float(-val)
                            have = agent.cargo.get(r.resource_id, 0.0)
                            drop = min(want_drop, have)

                            if drop > 0 and self.map_depots[px, py] == r.resource_id:
                                for depot in self.possible_depots:
                                    if depot.resource_id == r.resource_id and np.all(depot.position == [px, py]):
                                        depot.stock += drop
                                        break
                                agent.cargo[r.resource_id] -= drop
                                # Record the resources deposited.
                                resources_deposited[agent][r] += drop

        # Check termination conditions.
        end_truncate = lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles)
        end_done = False  # No done conditions for now.

        # Perform post-step calculations.
        for agent in self.possible_agents:
            # Perform observation.
            agent_observation, obs_mask = self.observe(
                agent,
                senders = senders - {agent}
            )
            obs_dict[agent] = agent_observation
            info_dict[agent]["visibility_mask"] = obs_mask

            # Calculate reward.
            reward_dict[agent] += self.get_reward(agent, end_truncate, end_done, resources_deposited[agent])

        # Handle end of episode.
        if end_truncate or end_done:
            for agent in self.agents:
                info_dict[agent]["ready"] = True
                truncated_dict[agent] = True
            self.agents = []
        
        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, self.dones, truncated_dict, info_dict


    def get_reward(self, agent, end_truncate, end_done, resources_deposited=None):
        ''' Returns the reward for the given agent. '''
        
        reward = 0.0

        # TEST: Provide reward based on distance of agent to specific location.
        target_position = np.array([2.0, 2.0], dtype=np.float32)
        dist = np.linalg.norm(agent.position.astype(np.float32) - target_position)
        reward = -dist

        # # Reward for depositing resources.
        # if "resources_deposited" is not None:
        #     for r, amount in resources_deposited.items():
        #         reward += r.reward_deposit * amount

        return reward


    def _getAvailableActions(self, agent):
        ''' Returns the available actions for the given agent. '''

        return None