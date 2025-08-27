from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import functools
from gymnasium import spaces
import random
import numpy as np
from matplotlib import pyplot as plt
from copy import copy

from isru_zoo.env.entity import ENTITY_TYPE, AGENT_ROLE, CAP, Agent, Depot, Extractor, Hauler, Prospector
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
    parser.add_argument("--num_resources", type=int, default=20,
                        help="The number of resources to place in the world.")
    parser.add_argument("--randomize_num_resources", action="store_true",
                        help="Whether to randomize the number of resources at each reset up to `num_resources`.")
    parser.add_argument("--world_size", type=int, default=50,
                        help="The size of the world. The world is a square with side length `world_size`.")
    parser.add_argument("--observation_radius", type=int, default=10,
                        help="The radius within which agents can observe each other and resources.")
    parser.add_argument("--observation_mask", action="store_true",
                        help="Whether to apply visibility mask to returned observations.")
    parser.add_argument("--available_actions_mask", action="store_true",
                        help="Whether to return an available actions mask for each agent.")
    parser.add_argument("--hauler_capacity", type=float, default=10.0,
                        help="The maximum amount of resources a hauler can carry.")
    parser.add_argument("--hauler_pickup_threshold", type=float, default=1.5,
                        help="Max Euclidean distance (in grid units) a Hauler must be within of any Extractor to pick up resources.")
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
            num_resources: int = 20,
            randomize_num_resources: bool = False,
            world_size: int = 50,
            observation_radius: int = 10,
            observation_mask: bool = False,
            available_actions_mask: bool = False,
            hauler_capacity: float = 10.0,
            hauler_pickup_threshold: float = 1.5,
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
        self.randomize_num_resources = randomize_num_resources
        self.render_mode = render_mode
        self.mask_observations = observation_mask
        self.mask_available_actions = available_actions_mask
        self.default_observation_radius = observation_radius
        self.default_hauler_capacity = hauler_capacity
        self.hauler_pickup_threshold = hauler_pickup_threshold

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
            TestResource1(num_resources)
        ]

        # Set up depots.
        self.possible_depots = [
            Depot(
                position=self.get_random_position(),
                resource = r
            ) for r in self.possible_resources
        ]

        # Record the map shape.
        self.map_shape = (*self.world_dims, 2 * len(self.possible_resources) + 3)  # +3 for obstacles, agents, depots

        # Set up spaces.
        self.observation_spaces = spaces.Dict({
            agent: self.observation_space(agent) for agent in self.possible_agents
        })
        self.action_spaces = spaces.Dict({
            agent: self.action_space(agent) for agent in self.possible_agents
        })
        self.nearest_tile=None
        
        # Reset the environment.
        self.reset()


    def _find_extractor_over_resource(self, hauler_pos, resource):
        """Return (extractor, extractor_pos_int) if there exists an Extractor that is
        standing on a cell containing the given resource and is within the
        hauler_pickup_threshold of the given hauler position. Otherwise (None, None).
        """
        pos = np.asarray(hauler_pos, dtype=np.float32)
        
        for a in self.agents:
            if a.capabilities[CAP.EXTRACT]:
                ex_pos_int = a.position.astype(np.int32)
                # Extractor must be standing on a cell that actually has this resource
                if self.map_resources[resource][ex_pos_int[0], ex_pos_int[1]] > 0:
                    d = np.linalg.norm(pos - a.position)
                    if d <= self.hauler_pickup_threshold:
                        return a, ex_pos_int
        return None, None

        # Removed misplaced environment setup block from class scope.


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        if seed != None:
            random.seed(seed)

        # Reset resources.
        for r in self.possible_resources:
            if self.randomize_num_resources:
                r.reset(quantity=random.randint(1, r.quantity_max))
            else:
                r.reset()

        # Generate new map (obstacles, resources, etc.) and get available positions.
        positions_available = self.generate_map()        

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        for agent in self.agents:
            idx = np.random.randint(positions_available.shape[0])
            start_position = positions_available[idx]
            agent.reset(
                reset_start_position=True,
                position=start_position
            )
            if agent.capabilities[CAP.CARRY]:
                agent.cargo = {r.resource_id: 0.0 for r in self.possible_resources}

        # Reset other state.
        self.step_count = 0
        self.last_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.dones = dict.fromkeys(self.agents, False)

        info = {
            agent: {} for agent in self.agents
        }

        # Return the initial observation.
        observation = {}
        for agent in self.agents:
            obs, obs_mask = self.observe(agent)
            observation[agent] = obs
            info[agent]["visibility_mask"] = obs_mask

        return observation, info


    def generate_map(self):
        ''' Generates a random map for the environment. '''

        world_indices_x = np.arange(self.world_dims[0])
        world_indices_y = np.arange(self.world_dims[1])

        # Reset obstacles.
        self.map_obstacles = np.zeros(self.world_dims, dtype=np.float32)
        self.map_obstacles[
            np.random.choice(world_indices_x, self.num_obstacles),
            np.random.choice(world_indices_y, self.num_obstacles)
        ] = 1.0
        positions_available_mask = self.map_obstacles == 0
        positions_available = np.argwhere(positions_available_mask)

        # Reset depots.
        # Ensure they are placed in an available location.
        self.map_depots = np.zeros(self.world_dims, dtype=np.float32)
        for depot in self.possible_depots:
            idx = np.random.randint(positions_available.shape[0])
            depot.reset(
                reset_start_position=True,
                position=positions_available[idx]
            )
            self.map_depots[depot.position[0], depot.position[1]] = depot.resource_id
            positions_available_mask[depot.position[0], depot.position[1]] = False
        positions_available = np.argwhere(positions_available_mask)

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

        return positions_available


    def get_random_position(self):
        ''' Returns a random position in the world. '''

        return np.random.uniform(0, self.world_dims).astype(np.float32)


    def render(self, pred=None, figsize=(9, 6), history_length=2):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches.
                
            Returns:
                None
        '''

        # # Convert the predicted state back into a dictionary (unflatten).
        # pred_unflattened = []
        # pred_steps = pred.shape[0] if pred is not None else 0
        # for i in range(pred_steps):
        #     p = spaces.unflatten(self.observation_spaces, pred[i].flatten())
        #     pred_unflattened.append(p)

        # Plot state as a grid using matplotlib.
        plt.figure(figsize=figsize)

        # Plot the agents.
        for agent in self.agents:
            pos = agent.position.astype(np.int32)
            if agent.capabilities[CAP.CARRY]:
                plt.scatter(pos[0], pos[1], label=f"Hauler {self.possible_agents.index(agent)}", marker="^", s=100, alpha=0.5, color="red", edgecolor="black")
            elif agent.capabilities[CAP.EXTRACT]:
                plt.scatter(pos[0], pos[1], label=f"Extractor {self.possible_agents.index(agent)}", marker="o", s=100, alpha=0.5, color="yellow", edgecolor="black")
            elif agent.capabilities[CAP.PROSPECT]:
                plt.scatter(pos[0], pos[1], label=f"Prospector {self.possible_agents.index(agent)}", marker="*", s=100, alpha=0.5, color="green", edgecolor="black")

        # Plot the resources.
        for i, r in enumerate(self.possible_resources):
            color = plt.cm.get_cmap("tab10")(i)
            positions = np.argwhere(self.map_resources[r] > 0)
            if positions.size > 0:
                plt.scatter(positions[:, 0], positions[:, 1], label=f"Resource {r.resource_id}", alpha=0.5)
            
            # Plot depot for this resource type.
            depot_positions = np.argwhere(self.map_depots == r.resource_id)
            if depot_positions.size > 0:
                plt.scatter(depot_positions[:, 0], depot_positions[:, 1], label=f"Depot {r.resource_id}", marker="s", color=color, edgecolor="black", s=100)
                
        # Display the total reward for this step. Position this text below the subplots. Do not use suptitle.
        reward = sum(self.last_rewards.values())
        resources_deposited = sum(depot.stock for depot in self.possible_depots)
        resources_held = sum(sum(agent.cargo.values()) for agent in self.agents if agent.capabilities[CAP.CARRY])
        plt.figtext(0.5, 0.01, f"Step: {self.step_count}, Combined Step Reward: {reward:.2f}, Resources Deposited: {resources_deposited}, Resources Held: {resources_held}", ha="center", fontsize=8)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(-1, self.world_dims[0])
        plt.ylim(-1, self.world_dims[1])
        plt.gca().set_aspect('equal', adjustable='box')

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


    def render_OLD(self, pred=None, figsize=(9, 6), history_length=2):
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

        return self.render_state(state, figsize=figsize)


    def render_state(self, state, figsize=(9, 6)):
        ''' Renders the given state.
            
            Args:
                state (dict): The state to render.
                
            Returns:
                None or np.ndarray: None if render_mode is "human", otherwise an RGB array.
        '''

        # Plot state as a grid using matplotlib.
        plt.figure(figsize=figsize)

        # Plot the map layers.
        # map_layers = state["map"]
        map_layers = state
        num_layers = map_layers.shape[-1]
        for i in range(num_layers):
            plt.subplot(1, num_layers, i + 1)
            plt.imshow(map_layers[:, :, i], cmap="gray", origin="lower")
            plt.title(f"Layer {i}")
            plt.axis("off")
        
        # Display the total reward for this step. Position this text below the subplots. Do not use suptitle.
        reward = sum(self.last_rewards.values())
        plt.figtext(0.5, 0.01, f"Step: {self.step_count}, Combined Step Reward: {reward:.2f}", ha="center", fontsize=8)

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
        # return spaces.Dict({
        #     "id": spaces.Discrete(len(self.possible_agents)),
        #     "map": spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=self.map_shape,
        #         dtype=np.float32
        #     )
        # })

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.map_shape,
            dtype=np.float32
        )


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
                return spaces.MultiBinary(action_space.nvec)
            elif isinstance(action_space, spaces.Box) and np.issubdtype(action_space.dtype, np.integer):
                diff = action_space.high - action_space.low + 1
                return spaces.MultiBinary(diff)
            else:
                raise NotImplementedError(f"Action space {action_space} not supported for action masking.")
        return get_available_action_space(action_space)


    def state(self):
        ''' Returns the global state of the environment.
            This is useful for centralized training, decentralized execution. '''
        
        state = self._populateStateSpace(self.possible_agents[0], force_visible=True)[0]

        return state


    def observe(self, agent, senders=set()):
        ''' Returns the observation for the given agent.'''

        # Collect local data.
        local_obs, local_obs_mask = self._populateStateSpace(agent)

        return local_obs, local_obs_mask

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

        if self.mask_observations:
            result = combined_obs["map"] * combined_obs_mask["map"]
            return result, combined_obs_mask["map"]
        else:
            # return combined_obs, combined_obs_mask
            return combined_obs["map"], combined_obs_mask["map"]


    def available_actions(self, agent):
        ''' Returns the available actions for this agent. '''

        if not self.mask_available_actions:
            return None

        result = self.available_actions_space(agent).sample()
        
        # Prevent movement into obstacles or out of bounds.
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                vel = np.array([dx, dy], dtype=np.int32)
                pos = (agent.position + vel).astype(np.int32)
                idx = vel + 1  # Shift from [-1, 0, 1] to [0, 1, 2] for indexing
                if pos[0] < 0 or pos[0] >= self.world_dims[0] or \
                        pos[1] < 0 or pos[1] >= self.world_dims[1] or \
                        self.map_obstacles[pos[0], pos[1]] == 1:
                    # This movement would go out of bounds or into an obstacle; disable it.
                    result["movement"][idx[0], idx[1]] = 0
                else:
                    result["movement"][idx[0], idx[1]] = 1

        # Communication is always available.
        result["communication"] = np.ones_like(result["communication"])

        if agent.capabilities[CAP.CARRY]:
            # Hauler agents can always pick up or drop off resources.
            result["resources"] = np.ones_like(result["resources"])
        else:
            # Non-hauler agents cannot pick up or drop off resources.
            result["resources"] = np.zeros_like(result["resources"])
        
        result_flattened = spaces.flatten(self.available_actions_space(agent), result)
        return result_flattened


    def _populateStateSpace(self, agent, force_visible=False):
        ''' Returns a populated state/observation space.'''

        # Load agent data into a map.
        map_agents = np.zeros(self.world_dims, dtype=np.int32)
        for i, a in enumerate(self.agents):
            pos = a.position.astype(np.int32)
            map_agents[pos[0], pos[1]] = 1 + a.role.value
            if a == agent and not force_visible:
                # Distinguish the agent in the map.
                map_agents[pos[0], pos[1]] *= -1
        
        # Load agent cargo into the map.
        map_cargo = []
        for r in self.possible_resources:
            m = np.zeros(self.world_dims, dtype=np.float32)
            for a in self.agents:
                if a.capabilities[CAP.CARRY] and r.resource_id in a.cargo:
                    pos = a.position.astype(np.int32)
                    m[pos[0], pos[1]] = a.cargo[r.resource_id]
            map_cargo.append(m)

        # Build the combined map.
        layers = [self.map_obstacles, map_agents, self.map_depots, *self.map_resources.values(), *map_cargo]
        map_combined = np.stack(layers, axis=-1).astype(np.float32)

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
            if not agent.capabilities[CAP.PROSPECT]:
                # Mask out all resource layers for non-prospector local observations.
                # (Communicated observations will still be merged in `observe()`)
                obs_mask["map"][:, :, resource_layer:] = False

        # Update the discovered resources mask.
        if agent.capabilities[CAP.PROSPECT] and not force_visible:
            for i, r in enumerate(self.possible_resources):
                self.mask_map_resources_discovered |= (self.map_resources[r] > 0) & obs_mask["map"][:, :, resource_layer + i]
        
        return obs["map"], obs_mask["map"]
        # return obs, obs_mask


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
            "resources/discovered": 0.0,
            "resources/deposited": 0.0,
            "resources/step_picked_up": 0.0,
            "resources/step_dropped_off": 0.0,
            "resources/extant": 0.0,
            "resources/total": 0.0,
            "extractors/num_in_place": 0,
        }
        info_dict.update({agent: {} for agent in self.possible_agents})
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
                position_prev = np.copy(agent.position)
                agent.position = agent.position + agent.velocity
                pos_min = np.array([0.0, 0.0], dtype=np.float32)
                pos_max = self.world_dims.astype(np.float32) - 1.0
                agent.position = np.clip(agent.position, pos_min, pos_max)

                # Correct agent velocity to reflect actual movement (in case of collisions).
                agent.velocity = agent.position - position_prev

                # Handle communication.
                # if action["communication"][0] >= 0.5:
                #     senders.add(agent)

                # Corrected resource handling for Hauler agents
                if agent.capabilities[CAP.CARRY]:
                    action_vec = action["resources"]
                    px, py = agent.position.astype(np.int32)

                    capacity = agent.capabilities[CAP.CARRY_CAPACITY]
                    current_load = sum(agent.cargo.values())
                    free = max(0.0, capacity - current_load)

                    for idx, val in enumerate(action_vec):
                        if val > 0:
                            # PICKUP: Require an Extractor to be standing on a tile that contains
                            # this resource type, and the Hauler must be within pickup threshold
                            # of that Extractor. 
                            r = self.idx_to_res[idx]
                            extractor, ex_pos_int = self._find_extractor_over_resource(agent.position, r)
                            if extractor is None:
                                # No eligible extractor-on-resource in range; cannot pick up
                                # reward_dict[agent] -= 0.001
                                continue

                            # Do not allow pickup from depots.
                            if self.map_depots[ex_pos_int[0], ex_pos_int[1]] != 0:
                                continue

                            available = self.map_resources[r][ex_pos_int[0], ex_pos_int[1]]
                            want = float(val)
                            take = min(want, available, free)
                            if take > 0:
                                # Remove from the resource map at the extractor's tile
                                self.map_resources[r][ex_pos_int[0], ex_pos_int[1]] -= take
                                # Add to the hauler's cargo
                                reward_dict[agent]+= 100.0 * take
                                reward_dict[extractor] += 100.0 * take
                                agent.cargo[r.resource_id] = agent.cargo.get(r.resource_id, 0.0) + take
                                free -= take
                                info_dict["resources/step_picked_up"] += take
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
                                self.map_resources[r][px, py] += drop
                                reward_dict[agent] += 100.0 * depot.resource.reward_deposit * drop
                                info_dict["resources/step_dropped_off"] += drop
                elif agent.capabilities[CAP.EXTRACT]:
                    # Provide reward for Extractors that are sitting on a resource tile.
                    px, py = agent.position.astype(np.int32)
                    # Skip locations with depots.
                    if self.map_depots[px, py] == 0 :
                        # Check for resources at the extractor's position.
                        for r in self.possible_resources:
                            if self.map_resources[r][px, py] > 0:
                                # Extractor is sitting on a resource tile.
                                # reward_dict[agent] += r.reward_extraction
                                info_dict["extractors/num_in_place"] += 1

        # Calculate the percentage of resources deposited.
        total_resources = sum(r.quantity for r in self.possible_resources)
        deposited_resources = sum(depot.stock for depot in self.possible_depots)
        resource_deposit_percentage = deposited_resources / float(total_resources)

        # Check termination conditions.
        end_truncate = lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles)
        end_done = deposited_resources >= total_resources

        # Perform post-step calculations.
        for agent in self.possible_agents:
            # Calculate the previous number of resources discovered.
            resources_discovered_prev = self._get_num_resources_discovered()

            # Perform observation.
            agent_observation, obs_mask = self.observe(
                agent,
                senders = senders - {agent}
            )
            obs_dict[agent] = agent_observation
            info_dict[agent]["visibility_mask"] = obs_mask

            # Reward prospectors for new resources discovered.
            # if agent.capabilities[CAP.PROSPECT]:
            #     resources_discovered = self._get_num_resources_discovered()
            #     reward_dict[agent] += 1.0 * (resources_discovered - resources_discovered_prev)

            # Reward all agents for progress towards full resource deposit.
            # reward_dict[agent] += 5.0 * resource_deposit_percentage

            # Provide reward.
            reward_dict[agent] += self.reward(agent, end_truncate, end_done)

        # Handle end of episode.
        if end_truncate or end_done:
            for agent in self.agents:
                truncated_dict[agent] = True
            self.agents = []

        # Update information dictionary.
        info_dict["resources/discovered"] = self._get_num_resources_discovered()
        info_dict["resources/deposited"] = deposited_resources
        info_dict["resources/extant"] = total_resources - deposited_resources
        info_dict["resources/total"] = total_resources

        # Record last rewards.
        self.last_rewards = copy(reward_dict)

        return obs_dict, reward_dict, self.dones, truncated_dict, info_dict


    def reward(self, agent, end_truncate, end_done):
        ''' Returns the reward for the given agent. '''

        reward = 0.0

        if agent.role == AGENT_ROLE.HAULER:
            cargo_total = sum(agent.cargo.values())
            if cargo_total > 0:
                # Reward haulers for carrying resources.
                reward += 0.05 * cargo_total
                # Reward haulers for distance to the nearest depot.
                nearest_depot_dist = np.min([np.linalg.norm(agent.position - depot.position) for depot in self.possible_depots])
                reward += 1.0 / (1.0 + nearest_depot_dist)

            # Reward haulers for distance to the nearest extractor.
            nearest_extractor_dist = self._nearest_extractor_distance(agent.position)
            reward += 1.0 / (1.0 + nearest_extractor_dist)

        elif agent.role == AGENT_ROLE.PROSPECTOR:
            # Reward prospectors for the number of resources discovered.
            num_discovered = self._get_num_resources_discovered()
            reward += 0.1 * num_discovered
        
        elif agent.role == AGENT_ROLE.EXTRACTOR:
            # Reward extractors for being on a resource tile.
            px, py = agent.position.astype(np.int32)
            if self.map_depots[px, py] == 0 :
                for r in self.possible_resources:
                    if self.map_resources[r][px, py] > 0:
                        reward += r.reward_extraction
                        break

        else:
            raise ValueError(f"Unknown agent role: {agent.role}")

        # Reward for global objective at the end of the episode.
        if True: #end_truncate or end_done:
            total_resources = sum(r.quantity for r in self.possible_resources)
            deposited_resources = sum(depot.stock for depot in self.possible_depots)
            resource_deposit_percentage = deposited_resources / float(total_resources)
            reward += 100.0 * resource_deposit_percentage

        return reward


    def get_nearest_uncleaned(self, agent):
        
        pos = agent.position
        # Get all indices where there has resources.
        indices = np.argwhere(self.map_resources == 1)
        if indices.size == 0:
            return np.array([0, 0], dtype=np.float32)
        
        # Compute differences from pos and squared distances (avoid sqrt for performance)
        diffs = indices - pos  # shape: (num_uncleaned, 2)
        squared_distances = np.sum(diffs**2, axis=1)
        
        # Find the index of the minimum squared distance
        min_idx = np.argmin(squared_distances)
        self.nearest_tile = indices[min_idx]
        
        # Return the relative difference as float32.
        return (self.nearest_tile - pos).astype(np.float32)


    def _get_num_resources_discovered(self):
        '''
        Returns the total number of resource cells discovered.
        '''
        
        mask = self.mask_map_resources_discovered
        return np.sum([np.sum(self.map_resources[r][mask]) for r in self.possible_resources])


    def _nearest_extractor_distance(self, position):
        """Return the minimum Euclidean distance from the given position to any Extractor.
        If no Extractor exists, return np.inf.
        """
        pos = np.asarray(position, dtype=np.float32)
        dmin = np.inf
        for a in self.agents:
            if a.capabilities[CAP.EXTRACT]:
                d = np.linalg.norm(pos - a.position)
                if d < dmin:
                    dmin = d
        return dmin


class parallel_env_simple_obs(parallel_env):
    ''' A simple observation version of the ISRU environment. '''

    @property
    @functools.cache
    def state_space(self):
        ''' Returns the state space of the environment. '''

        return spaces.Dict({
            "id": spaces.Box(
                low=0,
                high=len(self.possible_agents),
                dtype=np.int32,
            ),
            "agents": spaces.Dict({
                agent: spaces.Dict({
                    "position": spaces.Box(
                        low=0,
                        high=np.max(self.world_dims),
                        shape=(2,),
                        dtype=np.float32
                    ),
                    "velocity": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(2,),
                        dtype=np.float32
                    ),
                    "role": spaces.Box(
                        low=0,
                        high=len(AGENT_ROLE),
                        shape=(1,),
                        dtype=np.int32
                    ),
                    "cargo": spaces.Box(
                        low=0,
                        high=self.default_hauler_capacity,
                        shape=(len(self.possible_resources),),
                        dtype=np.float32
                    ),
                }) for agent in self.possible_agents
            }),
            "depots": spaces.Dict({
                depot: spaces.Dict({
                    "position": spaces.Box(
                        low=0,
                        high=np.max(self.world_dims),
                        shape=(2,),
                        dtype=np.float32
                    ),
                    "stock": spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(1,),
                        dtype=np.float32
                    ),
                }) for depot in self.possible_depots
            }),
            "resources": spaces.Dict({
                r: spaces.Box(
                    low=0,
                    high=np.max(self.world_dims),
                    shape=(r.quantity_max, len(self.world_dims)),
                    dtype=np.float32
                ) for r in self.possible_resources
            }),
        })


    def _populateStateSpace(self, agent, force_visible=False):
        ''' Fills in the state/observation space for the given agent. '''

        # Create the observation.
        obs = {
            "id": self.possible_agents.index(agent),
            "agents": {
                a: {
                    "position": a.position,
                    "velocity": a.velocity,
                    "role": np.array([a.role.value], dtype=np.int32),
                    "cargo": np.array(
                        [a.cargo.get(r.resource_id, 0.0) for r in self.possible_resources],
                        dtype=np.float32
                    )
                } for a in self.possible_agents
            },
            "depots": {
                d: {
                    "position": d.position,
                    "stock": np.array([d.stock], dtype=np.float32)
                } for d in self.possible_depots
            },
            "resources": {}
        }

        # Add resource positions.
        for r in self.possible_resources:
            obs["resources"][r] = np.ones((r.quantity_max, len(self.world_dims)), dtype=np.float32) * -1.0
            locations = np.argwhere(self.map_resources[r] > 0)
            obs["resources"][r][:locations.shape[0], :] = locations.astype(np.float32)

        obs_mask = {
            "id": np.ones_like(obs["id"], dtype=bool),
            "agents": {
                a: {
                    "position": np.ones_like(obs["agents"][a]["position"], dtype=bool),
                    "velocity": np.ones_like(obs["agents"][a]["velocity"], dtype=bool),
                    "role": np.ones_like(obs["agents"][a]["role"], dtype=bool),
                    "cargo": np.ones_like(obs["agents"][a]["cargo"], dtype=bool)
                } for a in self.possible_agents
            },
            "depots": {
                d: {
                    "position": np.ones_like(obs["depots"][d]["position"], dtype=bool),
                    "stock": np.ones_like(obs["depots"][d]["stock"], dtype=bool)
                } for d in self.possible_depots
            },
            "resources": {
                r: np.ones_like(obs["resources"][r], dtype=bool) for r in self.possible_resources
            }
        }

        return obs, obs_mask