from enum import IntEnum
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import functools
from gymnasium import spaces
import random
import numpy as np
from matplotlib import pyplot as plt
from copy import copy

from isru_zoo.env.entity import ENTITY_TYPE, AGENT_ROLE, CAP, Agent, Depot, Extractor, Hauler, Prospector, SuperBot
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
    parser.add_argument("--num_superbots", type=int, default=0,
                        help="The number of superbot vehicles to place in the world.")
    parser.add_argument("--num_obstacles", type=int, default=0,
                        help="The number of obstacles to place in the world.")
    parser.add_argument("--num_resources", type=int, default=20,
                        help="The number of resources to place in the world.")
    parser.add_argument("--randomize_num_resources", action="store_true",
                        help="Whether to randomize the number of resources at each reset up to `num_resources`.")
    parser.add_argument("--curriculum_num_resources", action="store_true",
                        help="Whether to increase the number of resources over time up to `num_resources`.")
    parser.add_argument("--world_size", type=int, default=50,
                        help="The size of the world. The world is a square with side length `world_size`.")
    parser.add_argument("--world_no_reset", action="store_true",
                        help="Whether to keep the same map between resets.")
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
        parsed_args.num_prospectors + \
        parsed_args.num_superbots


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
            num_superbots: int = 0,
            max_cycles: int = -1,
            episode_max: int = 1000,
            num_obstacles: int = 10,
            num_resources: int = 20,
            randomize_num_resources: bool = False,
            curriculum_num_resources: bool = False,
            world_size: int = 50,
            world_no_reset: bool = False,
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
        self.episode_max = episode_max
        self.world_dims = np.array([world_size, world_size], dtype=np.int32)
        self.world_no_reset = world_no_reset
        self.num_obstacles = num_obstacles
        self.randomize_num_resources = randomize_num_resources
        self.curriculum_num_resources = curriculum_num_resources
        self.render_mode = render_mode
        self.mask_observations = observation_mask
        self.mask_available_actions = available_actions_mask
        self.default_observation_radius = observation_radius
        self.default_hauler_capacity = hauler_capacity
        self.hauler_pickup_threshold = hauler_pickup_threshold
        self._pending_uncertainty = None  # (num_agents, H, W) or (num_agents, 1, H, W)

        # Set up entities.
        self.possible_agents = \
            [Extractor(
                world_dims=self.world_dims,
                position=self.get_random_position(),
                observation_radius=self.default_observation_radius
            ) for _ in range(num_extractors)] + \
            [Hauler(
                world_dims=self.world_dims,
                position=self.get_random_position(),
                carry_capacity=self.default_hauler_capacity,
                observation_radius=self.default_observation_radius
            ) for _ in range(num_haulers)] + \
            [Prospector(
                world_dims=self.world_dims,
                position=self.get_random_position(),
                observation_radius=self.default_observation_radius
            ) for _ in range(num_prospectors)] + \
            [SuperBot(
                world_dims=self.world_dims,
                position=self.get_random_position(),
                carry_capacity=self.default_hauler_capacity,
                observation_radius=self.default_observation_radius
            ) for _ in range(num_superbots)]

        # Set up the possible resources.
        self.possible_resources = []
        self.possible_resources.append(TestResource1(num_resources, resource_id=1))
        assert len(self.possible_resources) == 1, "only one resource type currently supported"

        # Set up depots.
        self.possible_depots = [
            Depot(
                position=self.get_random_position(),
                resource = r
            ) for r in self.possible_resources
        ]

        self.generated_map = None

        # Set up spaces.
        self.observation_spaces = spaces.Dict({
            agent: self.observation_space(agent) for agent in self.possible_agents
        })
        self.action_spaces = spaces.Dict({
            agent: self.action_space(agent) for agent in self.possible_agents
        })

        self.nearest_tile=None
        
        # Reset the environment.
        self.reset_count = 0
        self.reset()
    def set_uncertainty_channel(self, unc):
        """
        Accepts per-agent uncertainty for the NEXT observation.
        Shape: (num_agents, H, W) or (num_agents, 1, H, W).
        Agent order must match self.possible_agents.
        """
        self._pending_uncertainty = unc


    def _find_extractor_over_resource(self, hauler_pos, resource):
        """Return (extractor, extractor_pos_int) if there exists an Extractor that is
        standing on a cell containing the given resource and is within the
        hauler_pickup_threshold of the given hauler position. Otherwise (None, None).
        """
        pos = np.asarray(hauler_pos, dtype=np.float32)
        
        for a in self.agents:
            if a.capabilities[CAP.EXTRACT]:
                ex_pos_int = a.grid_position
                # Extractor must be standing on a cell that actually has this resource
                if self.map_resources[resource][ex_pos_int[0], ex_pos_int[1]] > 0:
                    d = np.linalg.norm(pos - a.position)
                    if d <= self.hauler_pickup_threshold:
                        return a, ex_pos_int
        return None, None



    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        if seed != None:
            np.random.seed(seed)

        self._pending_uncertainty = None
        
        # Reset resources.
        for r in self.possible_resources:
            r.reset()
        
        # Reset depots.
        for depot in self.possible_depots:
            depot.reset()

        # Generate new map (obstacles, resources, etc.) and get available positions.
        if self.generated_map is None or not self.world_no_reset:
            self.generated_map = self.generate_map()
        
        # Reset the map to match the generated map. Create a copy of the data since these variables will be modified.
        self.map_obstacles = self.generated_map[0].copy()
        self.map_depots = self.generated_map[1].copy()
        self.map_resources = {r: self.generated_map[2][r].copy() for r in self.possible_resources}
        positions_available = self.generated_map[3].copy()

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        for agent in self.agents:
            idx = np.random.randint(positions_available.shape[0])
            start_position = positions_available[idx]
            agent.reset(
                reset_start_position=True,
                resources=self.possible_resources,
                position=start_position,
            )

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
        
        self.reset_count += 1

        return observation, info


    def generate_map(self):
        ''' Generates a random map for the environment. '''

        mask_available = np.ones(self.world_dims, dtype=bool)

        # Reset obstacles.
        map_obstacles = np.zeros(self.world_dims, dtype=np.float32)
        positions_available = np.argwhere(mask_available)
        indices = np.random.choice(positions_available.shape[0], self.num_obstacles, replace=False)
        for i in indices:
            pos = positions_available[i]
            map_obstacles[pos[0], pos[1]] = 1.0
            mask_available[pos[0], pos[1]] = False

        # Reset depots.
        # Ensure they are placed in an available location.
        map_depots = np.zeros(self.world_dims, dtype=np.float32)
        positions_available = np.argwhere(mask_available)
        indices = np.random.choice(positions_available.shape[0], len(self.possible_depots), replace=False)
        for i, depot in enumerate(self.possible_depots):
            idx = indices[i]
            depot.reset(
                reset_start_position=True,
                position=positions_available[idx]
            )
            assert depot.resource_id != 0, "A resource_id of 0 is indistinguishable from empty space!"
            map_depots[depot.position[0], depot.position[1]] = depot.resource_id
            mask_available[depot.position[0], depot.position[1]] = False

        # Build stable resource index mappings for action vector <-> resource objects
        self.resource_list = list(self.possible_resources)
        self.rid_to_idx = {r.resource_id: i for i, r in enumerate(self.resource_list)}
        self.idx_to_res = {i: r for i, r in enumerate(self.resource_list)}

        # Reset resources.
        map_resources = {}
        for r in self.possible_resources:
            # Reset resource quantity if randomizing or using curriculum.
            if self.randomize_num_resources:
                r.reset(quantity=np.random.randint(1, r.quantity_max))
            elif self.curriculum_num_resources:
                # Scale the number of resources logarithmically with episode number.
                episode = self.reset_count / 2.0
                quantity = int(np.ceil((np.log1p(episode) / np.log1p(self.episode_max)) * r.quantity_max))
                r.reset(quantity=quantity)

            map_resources[r] = np.zeros(self.world_dims, dtype=np.float32)
            positions_available = np.argwhere(mask_available)
            indices = np.random.choice(positions_available.shape[0], r.quantity, replace=False)
            for i in range(r.quantity):
                idx = indices[i]
                pos = positions_available[idx]
                map_resources[r][pos[0], pos[1]] += 1.0
                mask_available[pos[0], pos[1]] = False

        # Ensure that the map was correctly generated.
        mask_resources = np.zeros(self.world_dims, dtype=bool)
        for r in self.possible_resources:
            mask_resources = mask_resources | (map_resources[r] > 0)
        mask_depots = map_depots > 0
        mask_obstacles = map_obstacles > 0
        assert np.sum(mask_resources & mask_obstacles) == 0, "Some resources are located inside obstacles!"
        assert np.sum(mask_resources & mask_depots) == 0, "Some resources are located inside depots!"
        assert np.sum(mask_depots & mask_obstacles) == 0, "Some depots are located inside obstacles!"
        assert np.sum(mask_resources) == sum(r.quantity for r in self.possible_resources), "Incorrect resource count!"
        assert np.sum(mask_depots) == len(self.possible_depots), "Incorrect depot count!"
        assert np.sum(mask_obstacles) == self.num_obstacles, "Incorrect obstacle count!"

        return map_obstacles, map_depots, map_resources, positions_available


    @property
    def map_shape(self):
        ''' Returns the map shape. '''
        return (2 * len(self.possible_resources) + 3, *self.world_dims)  # +3 for obstacles, agents, depots


    def get_random_position(self):
        ''' Returns a random position in the world. '''

        return np.random.uniform(0, self.world_dims).astype(np.float32)


    def render(self, pred=None, figsize=(9, 6), history_length=2, **kwargs):
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

        # Plot the depots.
        for i, r in enumerate(self.possible_resources):
            color = plt.cm.get_cmap("tab10")(i)
            # Plot depot for this resource type.
            depot_positions = np.argwhere(self.map_depots == r.resource_id)
            if depot_positions.size > 0:
                plt.scatter(depot_positions[:, 0], depot_positions[:, 1], label=f"Depot {r.resource_id}", marker="s", color=color, edgecolor="black", s=100)

        # Plot the agents.
        for agent in self.agents:
            pos = agent.grid_position
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

    def render_state(self, state, figsize=(9, 6)):
        ''' Renders the given state.
            
            Args:
                state (dict): The state to render.
                
            Returns:
                None or np.ndarray: None if render_mode is "human", otherwise an RGB array.
        '''

        from matplotlib.colors import ListedColormap
        import matplotlib
        MAP_LAYERS = parallel_env_map_obs.MAP_LAYERS

        # Plot state as a grid using matplotlib.
        plt.figure(figsize=figsize)

        # Plot the map layers.
        # map_layers = state["map"]
        num_layers = state.shape[0]

        # Create a gridspec with 2 rows: top row for layers, bottom row for human-readable
        import matplotlib.gridspec as gridspec
        fig = plt.gcf()
        fig.clf()
        gs = gridspec.GridSpec(2, num_layers, height_ratios=[1, 2])

        # Plot each layer in the top row
        for i in range(num_layers):
            ax = fig.add_subplot(gs[0, i])
            if i == MAP_LAYERS.RELATIVE_POS_X or i == MAP_LAYERS.RELATIVE_POS_Y:
                ax.imshow(np.abs(state[i, :, :]), cmap="coolwarm", vmin=-self.default_observation_radius, vmax=self.default_observation_radius)
            else:
                ax.imshow(state[i, :, :], cmap="plasma")
            name = MAP_LAYERS(i).name if i in MAP_LAYERS._value2member_map_ else f"Layer {i}"
            ax.set_title(name, fontsize=8, rotation=30)
            ax.axis("off")

        # Plot the human-readable version spanning the entire bottom row
        cmap_visibility = ListedColormap(['gray', 'white'])
        ax_hr = fig.add_subplot(gs[1, :])
        ax_hr.imshow(state[MAP_LAYERS.MASK_OBSERVED], cmap=cmap_visibility)
        ax_hr.set_title("Human-Readable State")
        ax_hr.axis("on")

        # Plot obstacles.
        positions = np.argwhere(state[MAP_LAYERS.OBSTACLES] > 0)
        ax_hr.scatter(positions[:, 1], positions[:, 0], label="Obstacle", marker="X", color="black", s=100)

        # Plot visible depots.
        positions = np.argwhere(state[MAP_LAYERS.DEPOTS] > 0)
        ax_hr.scatter(positions[:, 1], positions[:, 0], label="Depot", marker="$\u2302$", color="cyan", s=100)

        # Plot visible resources.
        positions = np.argwhere(state[MAP_LAYERS.RESOURCES_EXTANT] > 0)
        if positions.size > 0:
            ax_hr.scatter(positions[:, 1], positions[:, 0], marker="o", label=f"Resource", alpha=0.5)
        
        # Plot prospectors.
        positions = np.argwhere(state[MAP_LAYERS.AGENTS_PROSPECTOR] > 0)
        if positions.size > 0:
            ax_hr.scatter(positions[:, 1], positions[:, 0], label="Prospector", marker="$🔍$", s=100, alpha=0.5, color="green", edgecolor="black")
        
        # Plot extractors.
        positions = np.argwhere(state[MAP_LAYERS.AGENTS_EXTRACTOR] > 0)
        if positions.size > 0:
            ax_hr.scatter(positions[:, 1], positions[:, 0], label="Extractor", marker="$\u26CF$", s=100, alpha=0.5, color="yellow", edgecolor="black")

        # Plot haulers.
        positions = np.argwhere(state[MAP_LAYERS.AGENTS_HAULER] > 0)
        if positions.size > 0:
            ax_hr.scatter(positions[:, 1], positions[:, 0], label="Hauler", marker="$🚘$", s=100, alpha=0.5, color="red", edgecolor="black")

        # Plot this agent as a square with a black border around the original icon.
        # Position is where MAP_LAYERS.RELATIVE_POS_X and MAP_LAYERS.RELATIVE_POS_Y are both 0.
        where_x_zero = state[MAP_LAYERS.RELATIVE_POS_X] == 0
        where_y_zero = state[MAP_LAYERS.RELATIVE_POS_Y] == 0
        positions = np.argwhere(where_x_zero & where_y_zero)
        if positions.size > 0:
            ax_hr.scatter(positions[:, 1], positions[:, 0], label="Self", marker="s", s=150, facecolors='none', edgecolors='black', linewidths=2)

        # Plot a partially-completed ring around the haulers to indicate their cargo.
        cargo_layer = MAP_LAYERS.RESOURCES_CARGO
        positions = np.argwhere(state[cargo_layer] > 0)
        for pos in positions:
            cargo_percentage = state[cargo_layer][pos[0], pos[1]]
            if cargo_percentage > 0:
                # Draw an arc to indicate the amount of cargo.
                arc = matplotlib.patches.Arc(
                    (pos[1], pos[0]), 1.5, 1.5,
                    angle=0,
                    theta1=0,
                    theta2=cargo_percentage * 360,
                    color="red",
                    lw=2
                )
                ax_hr.add_patch(arc)

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

        return self.observation_space(self.possible_agents[0])


    @functools.cache
    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''

        def agent_obs_space(agent):
            ''' Returns the observation space for a single agent. '''

            return spaces.Dict({
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
            })

        # Set up agent state spaces. Ensure that each agent's data is first for its own observation.
        # Python dictionaries maintain insertion order as of Python 3.7.
        agent_spaces = {}
        agent_spaces[agent] = agent_obs_space(agent)
        for other_agent in self.possible_agents:
            if other_agent != agent:
                agent_spaces[other_agent] = agent_obs_space(other_agent)

        # Create the state space.
        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        return spaces.Dict({
            "role": spaces.Box(
                low=0,
                high=len(AGENT_ROLE),
                shape=(1,),
                dtype=np.int32
            ),
            "agents": spaces.Dict(agent_spaces),
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
            "map": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(*self.world_dims, len(self.possible_resources)),
                dtype=np.float32
            )
        })

        # return spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=self.map_shape,
        #     dtype=np.float32
        # )


    @functools.cache
    def action_space(self, agent):
        ''' Returns the action space for the given agent. '''
        
        return spaces.Dict({
            # Movement is specified in terms of the Moore neighborhood.
            # The agent can only move one space at a time.
            "movement": spaces.Box(low=0, high=8, shape=(1,), dtype=np.int32),

            # Communication is a request flag (request-based comm).
            "communication": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),

            # Resource actions are represented as follows:
            # -1 = drop off all resources
            # 0 = do nothing
            # 1 = pick up resources until full or no more available
            "resources": spaces.Box(low=-1, high=1, shape=(len(self.possible_resources),), dtype=np.int32),
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
        
        state = self._state()

        return state


    def _state(self):
        ''' Returns the global state and mask of the environment.'''

        return self._observe(self.possible_agents[0], force_visible=True)[0]


    def observe(self, agent, senders=set()):
        ''' Returns the observation for the given agent.'''

        # Collect local data.
        local_obs, local_obs_mask = self._observe(agent)

        return local_obs, local_obs_mask


    def available_actions(self, agent):
        ''' Returns the available actions for this agent. '''

        if not self.mask_available_actions:
            return None

        result = self.available_actions_space(agent).sample()

        other_agent_pos = [a.position for a in self.agents if a != agent]
        
        # Prevent movement into obstacles or out of bounds or into another agent.
        for move in range(9):
            velocity = self._moore_index_to_velocity(move)
            new_pos = agent.position + velocity
            new_pos_int = new_pos.astype(np.int32)
            if np.any(new_pos < 0) or np.any(new_pos >= self.world_dims):
                # Out of bounds.
                result["movement"][move] = 0
            elif self.map_obstacles[new_pos_int[0], new_pos_int[1]] > 0:
                # Obstacle in the way.
                result["movement"][move] = 0
            elif any(np.array_equal(new_pos_int, oap.astype(np.int32)) for oap in other_agent_pos):
                # Another agent is in the way.
                result["movement"][move] = 0
            else:
                result["movement"][move] = 1

        # Communication is always available.
        result["communication"] = np.ones_like(result["communication"])

        if agent.capabilities[CAP.CARRY]:
            for i, r in enumerate(self.possible_resources):
                # Can only drop off resources if at a depot and carrying some.
                px, py = agent.grid_position
                at_depot = self.map_depots[px, py] == r.resource_id
                is_carrying = agent.cargo.get(r.resource_id, 0) > 0
                result["resources"][0 + i] = at_depot and is_carrying
                
                # Can always do nothing.
                result["resources"][1 + i] = 1  # Do nothing

                # Can only pick up resources if near an extractor that is standing on a resource.
                extractor, ex_pos_int = self._find_extractor_over_resource(agent.position, r)
                has_capacity = agent.cargo[r.resource_id] < agent.capabilities[CAP.CARRY_CAPACITY]
                result["resources"][2 + i] = (extractor is not None) and has_capacity
        else:
            # Non-hauler agents cannot pick up or drop off resources.
            result["resources"] = np.zeros_like(result["resources"])
        
        result_flattened = spaces.flatten(self.available_actions_space(agent), result)
        return result_flattened


    def _observe(self, agent, force_visible=False):
        ''' Returns a populated state/observation space.'''

        # Build the combined map.
        layers = [*self.map_resources.values()]
        map_combined = np.stack(layers, axis=-1).astype(np.float32)

        # Create the observation.
        obs = {
            "role": np.array([agent.role.value], dtype=np.int32),
            "agents": {},
            "depots": {
                d: {
                    "position": d.position.astype(np.float32),
                    "stock": np.array([d.stock], dtype=np.float32),
                } for d in self.possible_depots
            },
            "map": map_combined
        }

        def get_agent_state(a):
            return {
                "position": a.position.astype(np.float32),
                "velocity": a.velocity.astype(np.float32),
                "role": np.array([a.role.value], dtype=np.int32),
                "cargo": np.array(
                    [a.cargo.get(r.resource_id, 0.0) for r in self.possible_resources],
                    dtype=np.float32
                ) / a.capabilities.get(CAP.CARRY_CAPACITY, 1.0),  # Normalize cargo by capacity
            }

        # Insert the ego agent first. Python dictionaries preserve insertion order.
        obs["agents"][agent] = get_agent_state(agent)

        # Add the rest of the agents.
        for a in self.possible_agents:
            if a != agent:
                obs["agents"][a] = get_agent_state(a)

        obs_mask = {
            "role": np.ones_like(obs["role"], dtype=bool),
            "agents": {a: {k: np.ones_like(v, dtype=bool) for k, v in adict.items()} for a, adict in obs["agents"].items()},
            "depots": {d: {k: np.ones_like(v, dtype=bool) for k, v in ddict.items()} for d, ddict in obs["depots"].items()},
            "map": np.ones_like(obs["map"], dtype=bool),
        }
        
        return obs, obs_mask


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

        # Reward constants.
        REWARD_COLLISION = -2.0
        REWARD_NO_EXPLORATION = -1.0
        REWARD_EXTRACTOR_ON_RESOURCE = 20.0
        REWARD_DEPOSIT = 0.0
        REWARD_EXTRACT = 100000.0
        REWARD_CLOSEST_RESOURCE = 0.2
        REWARD_DONE = 1000000.0

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
            "communication/requests_made": 0,
        }
        info_dict.update({agent: {} for agent in self.possible_agents})
        requesters = set()
        stack_value = {a: True for a in self.possible_agents}

        # Pre-movement calculations.
        visible_cells_prev = {}
        nearest_resource_dist_prev = {}
        for agent in self.agents:
            # Visible cells.
            visible_cells_prev[agent] = self._get_visible_cell_count(agent)

            # Nearest resource distance.
            resource_pos = self._get_nearest_resource(agent.position)
            if resource_pos is not None:
                nearest_resource_dist_prev[agent] = np.linalg.norm(agent.position - resource_pos)
        
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
                agent.velocity = self._moore_index_to_velocity(action["movement"][0])
                agent.velocity = np.clip(agent.velocity, -1.0, 1.0)

                # Move the agent.
                position_prev = np.copy(agent.position)
                agent.position = agent.position + agent.velocity
                pos_min = np.array([0.0, 0.0], dtype=np.float32)
                pos_max = self.world_dims.astype(np.float32) - 1.0
                agent.position = np.clip(agent.position, pos_min, pos_max)

                # Correct agent velocity to reflect actual movement (in case of collisions).
                if not np.allclose(agent.position - position_prev, agent.velocity):
                    agent.velocity = agent.position - position_prev
                    reward_dict[agent] += REWARD_COLLISION
                    stack_value[agent] = False

                # Check for how long the agent has been stationary.
                if np.linalg.norm(agent.velocity) < 1e-5:
                    agent.steps_stationary += 1
                else:
                    agent.steps_stationary = 1

                # Handle communication (request-based): agent requests others' observations.
                if True: #action["communication"][0] >= 0.5:
                    requesters.add(agent)
                    info_dict["communication/requests_made"] += 1

                # Corrected resource handling for Hauler agents
                if agent.capabilities[CAP.CARRY]:
                    action_vec = action["resources"]
                    px, py = agent.grid_position

                    capacity = agent.capabilities[CAP.CARRY_CAPACITY]
                    current_load = sum(agent.cargo.values())
                    free = max(0.0, capacity - current_load)

                    for idx, val in enumerate(action_vec):
                        # Just pickup/drop all possible.
                        val = val * capacity

                        if val > 0:
                            # PICKUP: Require an Extractor to be standing on a tile that contains
                            # this resource type, and the Hauler must be within pickup threshold
                            # of that Extractor. 
                            r = self.idx_to_res[idx]
                            extractor, ex_pos_int = self._find_extractor_over_resource(agent.position, r)
                            if extractor is None:
                                # No eligible extractor-on-resource in range; cannot pick up
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
                                agent.cargo[r.resource_id] = agent.cargo.get(r.resource_id, 0.0) + take
                                info_dict["resources/step_picked_up"] += take
                                reward_dict[agent] += REWARD_EXTRACT
                        elif val < 0:
                            r = self.idx_to_res[idx]
                            want_drop = float(-val)
                            have = agent.cargo.get(r.resource_id, 0.0)
                            drop = min(want_drop, have)

                            if drop > 0 and self.map_depots[px, py] == r.resource_id:
                                depot = self.possible_depots[idx]
                                depot.stock += drop
                                agent.cargo[r.resource_id] -= drop
                                info_dict["resources/step_dropped_off"] += drop
                                reward_dict[agent] += REWARD_DEPOSIT

                if agent.capabilities[CAP.EXTRACT]:
                    # Provide reward for Extractors that are sitting on a resource tile.
                    px, py = agent.grid_position
                    # Skip locations with depots.
                    if self.map_depots[px, py] == 0 :
                        # Check for resources at the extractor's position.
                        for r in self.possible_resources:
                            if self.map_resources[r][px, py] > 0:
                                # Extractor is sitting on a resource tile.
                                info_dict["extractors/num_in_place"] += 1
                                reward_dict[agent] += REWARD_EXTRACTOR_ON_RESOURCE / agent.steps_stationary

        # Calculate the percentage of resources deposited.
        total_resources = sum(r.quantity for r in self.possible_resources)
        deposited_resources = sum(depot.stock for depot in self.possible_depots)
        held_resources = sum(sum(agent.cargo.values()) for agent in self.agents)
        resource_deposit_percentage = deposited_resources / float(total_resources)

        # Check termination conditions.
        end_truncate = lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles)
        end_done = deposited_resources >= total_resources

        # Perform post-step calculations.
        for agent in self.possible_agents:
            # Perform observation.
            requested = (agent in requesters)
            senders_set = (set(self.agents) - {agent}) if requested else set()
            agent_observation, obs_mask = self.observe(
                agent,
                senders = senders_set
            )
            obs_dict[agent] = agent_observation
            info_dict[agent]["visibility_mask"] = obs_mask
            # Log per-agent uncertainty summaries (if provided by runner this step)
            if self._pending_uncertainty is not None:
                try:
                    aidx = self.possible_agents.index(agent)
                    unc = np.asarray(self._pending_uncertainty[aidx], dtype=np.float32)
                    
                    if unc.ndim == 3 and unc.shape[0] == 1:
                        unc = unc[0]
                   
                    if unc.shape != tuple(self.world_dims):
                        try:
                            unc = unc.reshape(*self.world_dims)
                        except Exception:
                            pass
                    info_dict[agent]["uncertainty_mean"] = float(np.nanmean(unc))
                    info_dict[agent]["uncertainty_max"] = float(np.nanmax(unc))
                except Exception:
                    
                    pass

            # Check whether anything new was explored.
            visible_cells = self._get_visible_cell_count(agent)
            if visible_cells <= visible_cells_prev[agent]:
                reward_dict[agent] += REWARD_NO_EXPLORATION
                stack_value[agent] = False
            
            # Provide so-called "stacked value" for reward shaping.
            if stack_value[agent]:
                agent.stacked_value += 1
            else:
                agent.stacked_value = 0
            reward_dict[agent] += 1.5 ** agent.stacked_value

            # Provide reward shaping for moving closer to the nearest resource.
            nearest_resource = self._get_nearest_resource(agent.position)
            if nearest_resource is not None and agent in nearest_resource_dist_prev:
                nearest_resource_dist = np.linalg.norm(agent.position - nearest_resource)
                if nearest_resource_dist_prev[agent] > nearest_resource_dist:
                    reward_dict[agent] -= REWARD_CLOSEST_RESOURCE
                else:
                    reward_dict[agent] += REWARD_CLOSEST_RESOURCE
            
            # Provide a completion reward.
            if end_done:
                reward_dict[agent] += REWARD_DONE / self.step_count

        # Update information dictionary.
        info_dict["resources/discovered"] = self._get_num_extant_resources_discovered() + deposited_resources + held_resources
        info_dict["resources/deposited"] = deposited_resources
        info_dict["resources/deposited_percentage"] = resource_deposit_percentage
        info_dict["resources/extant"] = total_resources - deposited_resources
        info_dict["resources/total"] = total_resources
        for agent in self.possible_agents:
            info_dict[f"rewards/{agent}"] = reward_dict[agent]

        # Handle end of episode.
        # if end_truncate or end_done:
        if end_truncate:
            resources_held = sum(sum(agent.cargo.values()) for agent in self.agents)
            resources_map = sum(self.map_resources[r].sum() for r in self.possible_resources)
            resources_map_depots = sum(self.map_resources[r][self.map_depots > 0].sum() for r in self.possible_resources)

            # Check that everything looks good at the end of the episode.
            assert self._get_num_extant_resources_discovered() <= total_resources, "More resources discovered than available!"
            assert deposited_resources <= total_resources, "More resources deposited than available!"
            assert total_resources == deposited_resources + resources_held + resources_map - resources_map_depots, "Resource accounting error!"

            for agent in self.agents:
                truncated_dict[agent] = True
            self.agents = []

        # Record last rewards.
        self.last_rewards = copy(reward_dict)

        if np.any(np.isnan(list(reward_dict.values()))):
            raise ValueError("NaN detected in reward_dict!")

        self._pending_uncertainty = None
        return obs_dict, reward_dict, self.dones, truncated_dict, info_dict


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


    def _get_nearest_resource(self, position):
        '''
        Returns the nearest position of the given resource type from the specified position.
        If no such resource exists, returns None.
        '''
        
        pos = np.asarray(position, dtype=np.float32)
        dmin = np.inf
        nearest_pos = None

        for r in self.possible_resources:
            locations = np.argwhere(self.map_resources[r] > 0)
            for loc in locations:
                d = np.linalg.norm(pos - loc)
                if d < dmin:
                    dmin = d
                    nearest_pos = loc
        return nearest_pos


    def _get_visible_cell_count(self, agent, use_resource_mask=False):
        '''
        Returns the total number of cells seen by the given agent.
        '''
        if use_resource_mask:
            return np.sum(agent.mask_resources_observed)
        else:
            return np.sum(agent.mask_observed)


    def _get_num_extant_resources_discovered(self, mask=None):
        '''
        Returns the total number of resource cells discovered.
        '''
        if mask is None:
            mask = np.zeros(self.world_dims, dtype=bool)
            for agent in self.agents:
                mask |= agent.mask_resources_observed
        
        # Do not count resources that are in depots.
        mask &= (self.map_depots == 0)

        total = 0
        for r in self.possible_resources:
            total += np.sum((self.map_resources[r] > 0) & mask)
        return total


    def _nearest_entity_distance(self, position, entity_type=ENTITY_TYPE.AGENT, capability=None):
        """
        Return the minimum Euclidean distance from the given position to any agent with specified capability.
        If no such agent exists, return np.inf.
        """
        pos = np.asarray(position, dtype=np.float32)
        dmin = np.inf

        if entity_type == ENTITY_TYPE.AGENT:
            entities = self.agents
        elif entity_type == ENTITY_TYPE.DEPOT:
            entities = self.possible_depots
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        for e in entities:
            if capability == None or e.capabilities[capability]:
                d = np.linalg.norm(pos - e.position)
                if d < dmin:
                    dmin = d
        return dmin


    def _moore_index_to_velocity(self, index):
        ''' Converts a Moore neighborhood index (0-8) to a velocity vector. '''
        if index < 0 or index > 8:
            raise ValueError(f"Invalid Moore neighborhood index: {index}")
        dx = (index // 3) - 1
        dy = (index % 3) - 1
        return np.array([dx, dy], dtype=np.int32)


    def _velocity_to_moore_index(self, velocity):
        ''' Converts a velocity vector to a Moore neighborhood index (0-8). '''
        dx, dy = velocity
        if dx < -1 or dx > 1 or dy < -1 or dy > 1:
            raise ValueError(f"Invalid velocity for Moore neighborhood: {velocity}")
        return (dx + 1) * 3 + (dy + 1)


class parallel_env_simple_obs(parallel_env):
    ''' A simple observation version of the ISRU environment. '''

    @functools.cache
    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''

        return spaces.Dict({
            "agents": spaces.Dict({
                a: spaces.Dict({
                    "position": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
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
                }) for a in self.possible_agents
            }),
            "depots": spaces.Dict({
                depot: spaces.Dict({
                    "position": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
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
                    low=-np.inf,
                    high=np.inf,
                    shape=(r.quantity_max, len(self.world_dims)),
                    dtype=np.float32
                ) for r in self.possible_resources
            }),
        })


    def _observe(self, agent, force_visible=False):
        ''' Fills in the state/observation space for the given agent. '''

        def relative_position(pos):
            ''' Returns the position relative to the given agent. '''
            return (pos - agent.position).astype(np.float32)

        def agent_obs(a):
            return {
                "position": relative_position(a.position),
                "velocity": a.velocity,
                "role": np.array([a.role.value], dtype=np.int32),
                "cargo": np.array(
                    [a.cargo.get(r.resource_id, 0.0) for r in self.possible_resources],
                    dtype=np.float32
                ) / a.capabilities.get(CAP.CARRY_CAPACITY, 1.0),  # Normalize cargo by capacity
            }

        # Create the observation.
        obs = {
            "agents": {},
            "depots": {
                d: {
                    "position": relative_position(d.position),
                    "stock": np.array([d.stock], dtype=np.float32)
                } for d in self.possible_depots
            },
            "resources": {}
        }

        # Insert the ego agent first. Python dictionaries preserve insertion order.
        obs["agents"][agent] = agent_obs(agent)
        for a in self.possible_agents:
            if a != agent:
                obs["agents"][a] = agent_obs(a)

        # Add resource positions.
        for r in self.possible_resources:
            obs["resources"][r] = np.ones((r.quantity_max, len(self.world_dims)), dtype=np.float32) * -100.0
            # Get locations with resources that are not in depots.
            locations = np.argwhere(self.map_resources[r] > 0)
            if locations.shape[0] > 0:
                # Exclude locations that are in depots.
                depot_mask = self.map_depots[locations[:, 0], locations[:, 1]] == 0
                locations = locations[depot_mask]

                # Convert to relative positions.
                locations = relative_position(locations)

                # Sort locations by distance to the agent.
                dists = np.linalg.norm(locations, axis=1)
                sorted_indices = np.argsort(dists)
                locations = locations[sorted_indices]
            obs["resources"][r][:locations.shape[0], :] = locations.astype(np.float32)

        obs_mask = {
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


class parallel_env_map_obs(parallel_env):
    ''' A map-based observation version of the ISRU environment. Intended for use with CNNs. '''

    class MAP_LAYERS(IntEnum):
        OBSTACLES = 0
        AGENTS_PROSPECTOR = 1
        AGENTS_EXTRACTOR = 2
        AGENTS_HAULER = 3
        RELATIVE_POS_X = 4
        RELATIVE_POS_Y = 5
        DEPOTS = 6
        RESOURCES_EXTANT = 7
        RESOURCES_DEPOSITED = 8
        RESOURCES_CARGO = 9
        MASK_OBSERVED = 10
        MASK_RESOURCES_OBSERVED = 11
        UNCERTAINTY = 12

    @property
    def map_shape(self):
        ''' Returns the map shape. '''
        return (len(parallel_env_map_obs.MAP_LAYERS), *self.world_dims)


    @property
    @functools.cache
    def state_space_DISABLE(self):
        ''' Returns the state space of the environment. '''

        return parallel_env_simple_obs.observation_space(self, self.possible_agents[0])

        agent = self.possible_agents[0]

        def agent_obs_space(agent):
            ''' Returns the observation space for a single agent. '''

            return spaces.Dict({
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
            })

        # Set up agent state spaces. Ensure that each agent's data is first for its own observation.
        # Python dictionaries maintain insertion order as of Python 3.7.
        agent_spaces = {}
        agent_spaces[agent] = agent_obs_space(agent)
        for other_agent in self.possible_agents:
            if other_agent != agent:
                agent_spaces[other_agent] = agent_obs_space(other_agent)

        # Create the state space.
        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        return spaces.Dict({
            "role": spaces.Box(
                low=0,
                high=len(AGENT_ROLE),
                shape=(1,),
                dtype=np.int32
            ),
            "agents": spaces.Dict(agent_spaces),
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
            "resources_extant": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            "resources_deposited": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            "extractors_in_place": spaces.Box(
                low=0,
                high=len([a for a in self.possible_agents if a.capabilities[CAP.EXTRACT]]),
                shape=(1,),
                dtype=np.int32
            ),
        })

        # return spaces.Dict({
        #     "resources_deposited": spaces.Box(
        #         low=0,
        #         high=np.inf,
        #         shape=(1,),
        #         dtype=np.float32
        #     ),
        #     "extractors_in_place": spaces.Box(
        #         low=0,
        #         high=len([a for a in self.possible_agents if a.capabilities[CAP.EXTRACT]]),
        #         shape=(1,),
        #         dtype=np.int32
        #     ),
        # })
    

    def _state_DISABLE(self):
        ''' Returns the global state and mask of the environment.'''


        return parallel_env_simple_obs._observe(self, self.possible_agents[0], force_visible=True)[0]

        agent = self.possible_agents[0]
        force_visible = True

        # Create the observation.
        obs = {
            "role": np.array([agent.role.value], dtype=np.int32),
            "agents": {},
            "depots": {
                d: {
                    "position": d.position.astype(np.float32),
                    "stock": np.array([d.stock], dtype=np.float32),
                } for d in self.possible_depots
            },
            "resources_extant": np.array([sum(r.quantity for r in self.possible_resources) - sum(depot.stock for depot in self.possible_depots)], dtype=np.float32),
            "resources_deposited": np.array([sum(depot.stock for depot in self.possible_depots)], dtype=np.float32),
            "extractors_in_place": np.array([0], dtype=np.int32),
        }

        extractors_in_place = 0
        for a in self.agents:
            if a.capabilities[CAP.EXTRACT]:
                px, py = a.grid_position
                if self.map_depots[px, py] == 0 :
                    for r in self.possible_resources:
                        if self.map_resources[r][px, py] > 0:
                            extractors_in_place += 1
                            break
        obs["extractors_in_place"] = np.array([extractors_in_place], dtype=np.int32)

        def get_agent_state(a):
            return {
                "position": a.position.astype(np.float32),
                "velocity": a.velocity.astype(np.float32),
                "role": np.array([a.role.value], dtype=np.int32),
                "cargo": np.array(
                    [a.cargo.get(r.resource_id, 0.0) for r in self.possible_resources],
                    dtype=np.float32
                ) / a.capabilities.get(CAP.CARRY_CAPACITY, 1.0),  # Normalize cargo by capacity
            }

        # Insert the ego agent first. Python dictionaries preserve insertion order.
        obs["agents"][agent] = get_agent_state(agent)

        # Add the rest of the agents.
        for a in self.possible_agents:
            if a != agent:
                obs["agents"][a] = get_agent_state(a)

        return obs

        # def extractors_in_place():
        #     count = 0
        #     for a in self.agents:
        #         if a.capabilities[CAP.EXTRACT]:
        #             px, py = a.grid_position
        #             if self.map_depots[px, py] == 0 :
        #                 for r in self.possible_resources:
        #                     if self.map_resources[r][px, py] > 0:
        #                         count += 1
        #                         break
        #     return count


        # state = {
        #     "resources_deposited": np.array([sum(depot.stock for depot in self.possible_depots)], dtype=np.float32),
        #     "extractors_in_place": np.array([extractors_in_place()], dtype=np.int32),
        # }

        # return state


    @functools.cache
    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.map_shape,
            dtype=np.float32
        )


    def _observe(self, agent, force_visible=False):
        ''' Fills in the state/observation space for the given agent. '''

        layers = [None for _ in range(self.map_shape[0])]

        # Set up the agent maps.
        map_agents = np.zeros((*self.world_dims, len(AGENT_ROLE) + 1), dtype=np.int32)
        for i, a in enumerate(self.possible_agents):
            pos = a.grid_position
            map_agents[pos[0], pos[1], a.role.value] = 1
        
        # Set up the resource, cargo, and deposited resource maps.
        map_resources_extant = np.zeros(self.world_dims, dtype=np.float32)
        map_resources_deposited = np.zeros(self.world_dims, dtype=np.float32)
        map_resources_cargo = np.zeros(self.world_dims, dtype=np.float32)
        for r_idx, r in enumerate(self.possible_resources):
            map_resources_extant += (self.map_resources[r] > 0).astype(np.float32)
            for d in self.possible_depots:
                if d.resource == r:
                    pos = d.grid_position
                    map_resources_deposited[pos[0], pos[1]] = d.stock #/ r.quantity
        for a in self.agents:
            if a.capabilities[CAP.CARRY]:
                cargo_amount = sum(a.cargo.values())
                pos = a.grid_position
                map_resources_cargo[pos[0], pos[1]] += cargo_amount / a.capabilities[CAP.CARRY_CAPACITY]

        # Temporarily use a single boolean map for depots.
        map_depots = self.map_depots > 0
        assert len(self.possible_resources) == 1, "Currently only supports one resource type."
        
        # Set up relative position maps. They should be 0 at the agent position and increase by 1 for each cell away.
        map_rel_pos_x = np.zeros(self.world_dims, dtype=np.float32)
        map_rel_pos_x[:, :] = np.arange(self.world_dims[0], dtype=np.float32)[:, None] - agent.position[0]
        map_rel_pos_y = np.zeros(self.world_dims, dtype=np.float32)
        map_rel_pos_y[:, :] = np.arange(self.world_dims[1], dtype=np.float32)[None, :] - agent.position[1]
        
        # Set up the visibility mask.
        obs_mask = np.ones(self.map_shape, dtype=bool)

        # Calculate the visible area based on a circular observation radius.
        if not force_visible or True:
            radius = agent.observation_radius
            pos = agent.grid_position
            visible = (np.arange(self.world_dims[0])[:, None] - pos[0]) ** 2 + \
                (np.arange(self.world_dims[1])[None, :] - pos[1]) ** 2 <= radius ** 2
            obs_mask[:, ~visible] = False

        # Update visibility information.
        if not force_visible:
            # Update the agent's observed resources mask.
            if agent.capabilities[CAP.PROSPECT]:
                agent.mask_resources_observed[visible] = True
            else:
                # Resources can only be observed for the first time by a prospector.
                obs_mask[self.MAP_LAYERS.RESOURCES_EXTANT, ~agent.mask_resources_observed] = False
            
            # Update the agent's observed area mask.
            agent.mask_observed[visible] = True

        # The relative position layers are always visible.
        obs_mask[self.MAP_LAYERS.RELATIVE_POS_X] = True
        obs_mask[self.MAP_LAYERS.RELATIVE_POS_Y] = True

        # Obstacles are always visible once seen.
        obs_mask[self.MAP_LAYERS.OBSTACLES, agent.mask_observed] = True

        # Depots are always visible.
        obs_mask[self.MAP_LAYERS.DEPOTS] = True

        # The masks themselves are always visible.
        obs_mask[self.MAP_LAYERS.MASK_OBSERVED] = True
        obs_mask[self.MAP_LAYERS.MASK_RESOURCES_OBSERVED] = True

        # Load most map layers.
        layers[self.MAP_LAYERS.OBSTACLES] = self.map_obstacles
        layers[self.MAP_LAYERS.AGENTS_PROSPECTOR] = map_agents[:, :, AGENT_ROLE.PROSPECTOR.value]
        layers[self.MAP_LAYERS.AGENTS_EXTRACTOR] = map_agents[:, :, AGENT_ROLE.EXTRACTOR.value]
        layers[self.MAP_LAYERS.AGENTS_HAULER] = map_agents[:, :, AGENT_ROLE.HAULER.value]
        layers[self.MAP_LAYERS.RELATIVE_POS_X] = map_rel_pos_x
        layers[self.MAP_LAYERS.RELATIVE_POS_Y] = map_rel_pos_y
        layers[self.MAP_LAYERS.DEPOTS] = map_depots
        layers[self.MAP_LAYERS.RESOURCES_EXTANT] = map_resources_extant
        layers[self.MAP_LAYERS.RESOURCES_DEPOSITED] = map_resources_deposited
        layers[self.MAP_LAYERS.RESOURCES_CARGO] = map_resources_cargo
        layers[self.MAP_LAYERS.MASK_OBSERVED] = agent.mask_observed.astype(np.float32)
        layers[self.MAP_LAYERS.MASK_RESOURCES_OBSERVED] = agent.mask_resources_observed.astype(np.float32)

        # --- Uncertainty layer (per-agent, non-communicated) ---
        unc_layer = np.zeros(self.world_dims, dtype=np.float32)
        if self._pending_uncertainty is not None:
            # Extract this agent's index
            aidx = self.possible_agents.index(agent)
            unc = np.asarray(self._pending_uncertainty[aidx], dtype=np.float32)
            # Remove singleton dimension if present.
            if unc.ndim == 3 and unc.shape[0] == 1:
                unc = unc[0]
            
            if unc.shape != tuple(self.world_dims):
                try:
                    unc = unc.reshape(*self.world_dims)
                except Exception:
                    pass  
            unc_layer = unc

        layers[self.MAP_LAYERS.UNCERTAINTY] = unc_layer
        # Make uncertainty visible regardless of other masks
        obs_mask[self.MAP_LAYERS.UNCERTAINTY] = True

        # Build the combined map.
        obs = np.stack(layers, axis=0).astype(np.float32)
        
        return obs, obs_mask


    def observe(self, agent, senders=set()):
        ''' Returns the observation for the given agent.'''

        # Collect local data.
        local_obs, local_obs_mask = self._observe(agent)

        # Set up the matrices.
        map = np.copy(local_obs)
        map_mask = np.zeros_like(local_obs_mask, dtype=bool)

        # Handle communicated data.
        for sender in senders:
            sender_obs, sender_obs_mask = self._observe(sender)

            sender_visible = sender_obs_mask == True
            map[sender_visible] = sender_obs[sender_visible]
            map_mask[sender_visible] = True

            # Update the agent's known space masks.
            agent.mask_observed |= sender.mask_observed
            agent.mask_resources_observed |= sender.mask_resources_observed
        
        # Apply local observations (overwrite any communicated data).
        map[local_obs_mask == True] = local_obs[local_obs_mask == True]
        map_mask |= local_obs_mask

        # Apply the local relative position layers (overwrite any communicated data).
        map[self.MAP_LAYERS.RELATIVE_POS_X] = local_obs[self.MAP_LAYERS.RELATIVE_POS_X]
        map[self.MAP_LAYERS.RELATIVE_POS_Y] = local_obs[self.MAP_LAYERS.RELATIVE_POS_Y]
        map_mask[self.MAP_LAYERS.RELATIVE_POS_X] = True
        map_mask[self.MAP_LAYERS.RELATIVE_POS_Y] = True

        # Update the local observation.
        combined_obs = map
        combined_obs_mask = map_mask

        if self.mask_observations:
            result = combined_obs * combined_obs_mask
            return result, combined_obs_mask
        else:
            # return combined_obs, combined_obs_mask
            return combined_obs, combined_obs_mask

class parallel_env_flat_map_obs(parallel_env):
    ''' A single-layer map-based observation of the ISRU environment. '''

    @functools.cache
    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(self.world_dims) + 1 + len(self.possible_agents) * len(self.world_dims),),
            dtype=np.float32
        )


    def _observe(self, agent, force_visible=False):
        ''' Fills in the state/observation space for the given agent. '''

        def relative_position(pos):
            ''' Returns the position relative to the given agent. '''
            return (pos - agent.position).astype(np.float32)

        VALUE_OBSTACLE = -2.0
        VALUE_RESOURCE = 2.0
        VALUE_UNEXPLORED = 0.0
        VALUE_EXPLORED = -1.0

        # Create the map.
        map = np.ones(self.world_dims, dtype=np.float32) * VALUE_UNEXPLORED
        map[agent.mask_observed] = VALUE_EXPLORED
        map[self.map_obstacles > 0] = VALUE_OBSTACLE
        for r in self.possible_resources:
            map[self.map_resources[r] > 0] = VALUE_RESOURCE
        
        # Select the target.
        if np.any(map == VALUE_RESOURCE):
            target = self._get_nearest_resource(agent.position)
        else:
            target = self._nearest_entity_distance(agent.position, entity_type=ENTITY_TYPE.DEPOT)

        # Collect other components. 
        agent_id = agent.role.value * 10.0
        target_relative = relative_position(target)
        other_agents_pos = [relative_position(a.position) for a in self.possible_agents if a != agent]
        
        # Build the observation.
        obs = np.concatenate((
            np.array([agent_id], dtype=np.float32),
            target_relative.astype(np.float32),
            np.array(other_agents_pos, dtype=np.float32).flatten(),
            map.flatten(),
        ), axis=0)

        # Set up visibility mask.
        obs_mask = np.ones_like(obs, dtype=bool)

        return obs, obs_mask