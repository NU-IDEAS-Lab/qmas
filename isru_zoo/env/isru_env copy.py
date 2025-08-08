from pettingzoo import ParallelEnv
import numpy as np
from gymnasium.spaces import Box, Dict
from collections import deque
import matplotlib.pyplot as plt

def add_args(parser):
    ''' Adds environment arguments. '''
    
    pass
    # import argparse
    # parser.add_argument("--num_adversaries", type=int, default=1,
    #                     help="The number of adversaries in the environment.")
    # parser.add_argument("--num_dimensions", type=int, default=2,
    #                     help="The number of dimensions in the environment.")
    # parser.add_argument("--random_start_positions", action=argparse.BooleanOptionalAction, default=False,
    #                     help="If true, agents will start at random positions in the world. If false, they will start at [0,0].")
    # parser.add_argument("--state_per_agent", action=argparse.BooleanOptionalAction, default=False,
    #                     help="If true, the state function will return a separate copy of the state for each agent. "
    #                          "This is useful for centralized training, decentralized execution. "
    #                          "If false, the state function will return a single copy of the state that is shared among all agents.")
    # parser.add_argument("--observation_probability", type=float, default=1.0,
    #                     help="The probability that an agent will observe another entity.")

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
    }

    def __init__(self, agent_count=1, world_size=(20,10),timestep=0.8,partial_view_size=(50, 5)):
        """
        Initialize the environment.

        :param agent_count: Number of agents in the environment.
        :param world_size: Size of the world (width, height).
        :param timestep: Time step for the simulation.
        """
        self.agent_count = agent_count
        self.world_size = world_size
        self.timestep = timestep
        self.step_count=0
        self.base_info=0
        self.partial_view_size = partial_view_size
        self.possible_agents = [f"agent_{i+1}" for i in range(agent_count)]
        self.shape=8 + 2
        # Initialize observation and action spaces
        self.observation_spaces = Dict({
            f"agent_{i+1}": Box(low=0, high=255, shape=(self.shape,), dtype=np.float32)
            for i in range(agent_count)
        })
        self.state_space=Dict({
            f"agent_{i+1}": Box(low=0, high=255,  shape=(self.shape,), dtype=np.float32)
            for i in range(agent_count)
        })
        self.action_spaces = Dict({
            f"agent_{i+1}": Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            for i in range(agent_count)
        })

        # Internal state
        self.observations = {agent: None for agent in self.possible_agents}
        self.entrance_x = 0
        self.entrance_y = world_size[1] // 2
        self.positions = {
            f"agent_{i+1}": np.array([self.entrance_x + i, self.entrance_y]) for i in range(agent_count)
        }  # Agents queued at entrance
        self.unvisited_value=0
        self.omv=-2
        self.unvisited = np.full(world_size,self.unvisited_value,dtype=np.float32)  # Track visited positions              # unvisited is 1
        self.obstacles = np.random.choice([self.omv, 0], size=(int(world_size[0]), int(world_size[1])), p=[0.01, 0.99])       
        self.obstacle_value=self.unvisited_value+self.omv                                                                 # obstacle is 2
        # self.resources = np.random.choice([0.5, 0], size=(int(world_size[0]), int(world_size[1])), p=[0.1, 0.9]) # resource is 1.5
        self.map=self.unvisited+self.obstacles
        self.power=300
        self.power_levels = {f"agent_{i+1}":self.power for i in range(agent_count)}
        self.storage_capacity = {f"agent_{i+1}": 50 for i in range(agent_count)}
        self.dones = {agent: False for agent in self.positions.keys()}
        self.truncs = {agent: False for agent in self.positions.keys()}
        # Ensure the agent won't overlap with rewards and obstacles at the beginning
        for i in range(agent_count):
            self.map[self.entrance_x + i, self.entrance_y] = self.unvisited_value

        # Communication part
        self.communication_ranges = {f"agent_{i+1}": np.random.uniform(9.0, 11.0) for i in range(agent_count)}
        self.data_collected = {f"agent_{i+1}": 0 for i in range(agent_count)}  # Data collected by each agent
        self.communication_graph = {}
        self.trajectories = {agent: [] for agent in self.possible_agents}
        # === New: Reward Shaping with Stacked Value ===
        # For each agent, we maintain a "stacked value" counter.
        self.stacked_values = {agent: 0 for agent in self.possible_agents}
        
        # To track cumulative rewards per episode (used for elite set selection)
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        
        # === New: Elite Set ===
        # A collection to store high-performance episodes.
        self.elite_set = []
        obstacle_proportion = self.calculate_obstacle_proportion()
        print(f"Obstacle Proportion: {obstacle_proportion:.2f}")
        self.render_train()

    def step(self, actions, lastStep=False):
        """
        Perform a step in the environment.

        :param actions: Dictionary of agent actions.
        """
    
        infos = {agent: {0} for agent in self.positions.keys()}
        rewards = {agent: 0.0 for agent in self.positions.keys()}
        steps=200
        if self.step_count % steps == 0 and self.step_count>0:
            self.plot_interval_trajectories(self.step_count - steps, self.step_count)

        
        for agent, position in self.positions.items():
            self.trajectories[agent].append(position.copy())

        distances = {
            agent: np.linalg.norm(self.positions[agent] - np.array([self.entrance_x, self.entrance_y]))
            for agent in self.positions.keys()
        }

        relaying_agent = min(distances, key=distances.get)

        # if  exploration_proportion-infoCollected_proportion>0.2:
        #     rewards[relaying_agent]-=100
        self.step_count +=1 
    
        for agent, action in actions.items():
            if self.power_levels[agent] == 0:
                self.dones[agent]=True
                continue

            movement_cost = 0.1
            self.power_levels[agent] -= movement_cost
            self.power_levels[agent] = max(0, self.power_levels[agent])

            prev_x, prev_y = int(self.positions[agent][0]), int(self.positions[agent][1])

            raw_position= self.positions[agent] + np.clip(action, -1, 1) * self.timestep
            new_position= np.clip(raw_position, 0, [self.world_size[0] - 1, self.world_size[1] - 1])
            if not np.array_equal(raw_position, new_position) or self.map[int(new_position[0]), int(new_position[1])]==-2 :
                rewards[agent] -= 2  # Penalty for hitting the wall

            # Ensure the agent does NOT move into an obstacle and does not move into an occupied space
            # if self.obstacles[int(new_pos[0]), int(new_pos[1])] == 1 :
            #     deviation = np.random.uniform(-1, 1, size=2) * self.timestep
            #     alternative_position = np.clip(self.positions[agent] + deviation, 0, [self.world_size[0] - 1, self.world_size[1] - 1])
            
            #     if self.obstacles[int(alternative_position[0]), int(alternative_position[1])]==0:
            #         self.positions[agent] = alternative_position
            
            #     rewards[agent] -= 0.05  # Penalty for hitting an obstacle
            else:
                self.positions[agent] = new_position                

            grid_x, grid_y = int(self.positions[agent][0]), int(self.positions[agent][1])
            exploration_reward=0
            # Check if this cell was visited before
            if self.map[grid_x, grid_y] == self.unvisited_value:     # if it is equal to zero
                self.map[grid_x, grid_y] = -1 # Mark as visited
            
                self.data_collected[agent] += 1
            elif self.map[grid_x, grid_y]==-1:
                rewards[agent]-=1
            
            # === New: Apply Reward Shaping using Stacked Value ===
            # If the immediate reward is >= 0, increase the stacked value and compute bonus.
            # If negative, reset the stacked value.
            if rewards[agent] >= 0:              # in this case there is no positive reward
                self.stacked_values[agent] += 1
                bonus = 1.5 ** self.stacked_values[agent]
            else:
                self.stacked_values[agent] = 0
                bonus = 1.5 ** 0  # equals 1
            rewards[agent] += bonus

            # Update cumulative episode rewards
            self.episode_rewards[agent] += rewards[agent]

            # Check for resource collection
            # if self.resources[grid_x, grid_y] == 285:
            #     self.resources[grid_x, grid_y] = 0
            #     resource_bonus = 5

            # distance_to_start = np.linalg.norm(self.positions[agent] -[0,self.entrance_y])
            # continuous_reward = 0.05 * distance_to_start
            # stagnation_penalty = -2 if exploration_reward == 0 else 0

            rewards[agent]+= exploration_reward

        
        if self.step_count >= 100:
            # trunc the agent
            self.truncs = {agent: True for agent in self.possible_agents}

        if all(self.dones.values()):
            self.plot_interval_trajectories(0, self.step_count)
       
        
        
        return  {agent: self.observe(agent) for agent in self.possible_agents}, rewards, self.dones, self.truncs,infos


    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """
        print(f"Step {self.step_count}: Exploration Proportion: {self.calculate_exploration_proportion():.2f}")
        print(f'Info Collected Proportion: {self.calculate_collectedinfo_proportion():.2f}')
        np.random.seed(seed)
        self.step_count=0
        self.positions = {
            f"agent_{i+1}": np.array([self.entrance_x + i, self.entrance_y]) for i in range(self.agent_count)
        }  # Reset agents queued at entrance
        self.unvisited = np.full(self.world_size,self.unvisited_value,dtype=np.float32)  # Track visited positions              # unvisited is 1
        self.obstacles = np.random.choice([self.omv, 0], size=(int(self.world_size[0]), int(self.world_size[1])), p=[0.05, 0.95])                                                                       # obstacle is 2
        # self.resources = np.random.choice([0.5, 0], size=(int(world_size[0]), int(world_size[1])), p=[0.1, 0.9]) # resource is 1.5
        self.map=self.unvisited+self.obstacles
        self.data_collected = {f"agent_{i+1}": 0 for i in range(self.agent_count)}
        self.power_levels = {f"agent_{i+1}": 100.0 for i in range(self.agent_count)}
        self.rewards = {f"agent_{i+1}": 0 for i in range(self.agent_count)}
        self.base_info=0
        
        self.communication_graph = {}
        self.dones = {agent: False for agent in self.positions.keys()}
        self.truncs = {agent: False for agent in self.positions.keys()}
        self.trajectories = {agent: [] for agent in self.possible_agents}
        return {agent: self.observe(agent) for agent in self.possible_agents}, {}

    def observe(self, agent):
        x, y = map(int, self.positions[agent])
        
        # Offsets for the 8 surrounding tiles (Moore neighborhood)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0,  1),
            ( 1, -1), ( 1, 0), ( 1,  1)
        ]
        
        neighbor_values = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check bounds
            if 0 <= nx < self.world_size[0] and 0 <= ny < self.world_size[1]:
                neighbor_values.append(self.map[nx, ny])
            else:
                # Out of bounds treated as wall/obstacle
                neighbor_values.append(-2)

        rel_location = self.get_nearest_uncleaned(agent)
        obs = np.array(neighbor_values + list(rel_location), dtype=np.float32)
        obs=obs.reshape(1, self.shape)
        # print(obs.shape)
        return obs

    
    def get_nearest_uncleaned(self, agent):
        """
        Detects the nearest uncleaned tile (i.e. tile with value equal to self.unvisited_value)
        relative to the agent's current position. Returns a numpy array with the differences (Δx, Δy).
        If no uncleaned tile is found, returns [0, 0].
        """
        pos = self.positions[agent]
        min_distance = np.inf
        nearest_tile = None
        # Simple brute-force search over the grid
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                if self.map[i, j] == self.unvisited_value:
                    d = np.linalg.norm(np.array([i, j]) - pos)
                    if d < min_distance:
                        min_distance = d
                        nearest_tile = (i, j)
        if nearest_tile is not None:
            rel_x = nearest_tile[0] - pos[0]
            rel_y = nearest_tile[1] - pos[1]
            return np.array([rel_x, rel_y], dtype=np.float32)
        else:
            return np.array([0, 0], dtype=np.float32)
        
    # def relayAgent_reward(self,position,relayAgent):
    #     dis=(position[0]**2+position[1]**2)**1/2
    #     reward=-20*np.tanh(dis-self.communication_ranges[relayAgent]-50)
    #     return reward

    def update_communication_graph(self):
        """
        Build a one-directional communication graph based on communication ranges.
        """
        self.communication_graph = {agent: [] for agent in self.positions.keys()}
        for agent1, pos1 in self.positions.items():
            for agent2, pos2 in self.positions.items():
                if agent1 != agent2:
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance <= self.communication_ranges[agent1]:
                        self.communication_graph[agent1].append(agent2)

    def bfs_find_path(self, start, goal):
        """
        Perform BFS to find a path from start to goal in the communication graph.

        :param start: Starting agent.
        :param goal: Target agent.
        :return: List representing the path, or empty list if no path exists.
        """
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path

            visited.add(current)
            for neighbor in self.communication_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return []
    
    def state(self):
        return {a: self.observe(a) for a in self.possible_agents}
    
    def available_actions(self,agent):
        return None
    
    def calculate_collectedinfo_proportion(self):
        total_cells = self.unvisited.size
        return self.base_info/total_cells
    
    def calculate_obstacle_proportion(self):
        return np.sum(self.obstacles == self.omv) / np.prod(self.obstacles.shape)

    def calculate_exploration_proportion(self):
        visited = np.sum(self.map == -1)
        total = np.prod(self.map.shape)
        return visited / total
    
    def plot_interval_trajectories(self, start_step, end_step):
        """
        Plot the trajectories of agents during a specific interval.

        :param start_step: Start step of the interval.
        :param end_step: End step of the interval.
        """
        plt.figure(figsize=(10, 5))
        for agent, positions in self.trajectories.items():
            
            interval_positions = np.array(positions[start_step:end_step])
            plt.plot(interval_positions[:, 0], interval_positions[:, 1], linestyle='-', linewidth=1,label=agent)

        plt.title(f"Agent Trajectories from Step {start_step} to {end_step}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.savefig('Plot')
        plt.close()


    def render(self):
        """
        Render the complete trajectories of all agents.

        This function visualizes the movement paths of all agents, 
        marking the steps taken and highlighting the relaying agent.

        Called during test using: render_env.envs[0].env.render()
        """
        # print(self.calculate_exploration_proportion())
        plt.figure(figsize=(12, 6))
        
        # Identify the relaying agent (closest to the base station)
        distances = {
            agent: np.linalg.norm(self.positions[agent] - np.array([self.entrance_x, self.entrance_y]))
            for agent in self.positions.keys()
        }
        relaying_agent = min(distances, key=distances.get)  # Agent closest to the base

        for agent, positions in self.trajectories.items():
            if len(positions) > 1:
                trajectory = np.array(positions)

                # Plot trajectory
                if agent == relaying_agent:
                    plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='-', linewidth=2, label=f"Relaying Agent ({agent})", color='red')
                else:
                    plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='--', linewidth=1, label=f"Agent {agent}")

        # Mark base station position
        plt.scatter(self.entrance_x, self.entrance_y, marker='*', color='blue', s=100, label="Base Station")

        # Labels & Legend
        plt.title(f"Agent Trajectories (Total Steps: {self.step_count})")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.show()
        

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def render_train(self):
        plt.imshow(self.map, cmap="gray")
        plt.title("World Map")
        plt.show()




