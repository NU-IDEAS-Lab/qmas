from pettingzoo import ParallelEnv
import numpy as np
from gymnasium.spaces import Box, Dict, MultiDiscrete
from collections import deque
import matplotlib.pyplot as plt

class CustomEnvironment(ParallelEnv):
    metadata = {"name": "custom_environment_v0",}

    def __init__(self, agent_count=4, cave_size=(120, 40), timestep=0.4):
        """
        Initialize the environment.
        - Each agent starts with its own map which is a copy of the global map.
        - The global map is preloaded with obstacles (cells with value -2) and free space (0).
        - Power consumption, data collection, and reward shaping logic are incorporated.
        """
        self.agent_count = agent_count
        self.cave_size = cave_size
        self.timestep = timestep
        self.step_count = 0
        self.base_info = 0  # Total information delivered to the base.

        self.possible_agents = [f"agent_{i+1}" for i in range(agent_count)]
        self.half = 5
        # Observation consists of a 10x10 patch (100 values) plus 2 values for relative target location.
        self.shape = 10 * 10 + 2

        self.observation_spaces = Dict({
            f"agent_{i+1}": Box(low=-2, high=255, shape=(self.shape,), dtype=np.float32)
            for i in range(agent_count)
        })
        self.state_space = Box(low=-2, high=255, shape=(self.shape,), dtype=np.float32)
        action_space = Dict({
            MultiDiscrete([8, 2])})
        
        self.action_spaces=Dict({agent: action_space for agent in self.possible_agents})
        # Base (entrance) settings.
        self.entrance_x = 0
        self.entrance_y = cave_size[1] // 2
        self.positions = {
            f"agent_{i+1}": np.array([self.entrance_x + i, self.entrance_y], dtype=np.float32)
            for i in range(agent_count)
        }

        # Map values:
        # 0 represents free/unvisited.
        # -1 represents explored.
        # -2 represents an obstacle.
        self.unvisited_value = 0
        self.omv = -2  # obstacle marker
        self.res=1
        self.UNEXPLORED = 0
        # Obstacles are generated with a given probability.
        self.ospro = 0.1  # obstacle proportion
        self.obstacles = np.random.choice(
            [self.omv, 0, self.res],
            size=(int(cave_size[0]), int(cave_size[1])),
            p=[self.ospro, 0.8, 0.1]
        )
        self.nearest_unexplored_tile = None
        for i in range(agent_count):
            init_x = self.entrance_x + i
            init_y = self.entrance_y
            self.obstacles[init_x, init_y] = self.UNEXPLORED
        # Global map: initially a copy of obstacles (free space is 0, obstacles are -2).
        self.global_map = np.copy(self.obstacles)
        self.agini_map = np.copy(self.obstacles)
        for i in range(int(cave_size[0])):
            for j in range(int(cave_size[1])):
                if self.obstacles[i,j]==self.res:
                    self.agini_map[i,j] = self.unvisited_value
        # Each agent starts with its personal map equal to the global map.
        self.agent_maps = {
            agent: np.copy(self.agini_map)
            for agent in self.possible_agents
        }

        # ----- Power, Data, and Storage Settings -----
        self.power = 2000
        self.power_levels = {f"agent_{i+1}": self.power for i in range(agent_count)}
        self.storage_capacity = {f"agent_{i+1}": 5000000 for i in range(agent_count)}
        self.data_collected = {f"agent_{i+1}": 0 for i in range(agent_count)}

        self.dones = {agent: False for agent in self.positions.keys()}
        self.truncs = {agent: False for agent in self.positions.keys()}

        # Communication and trajectory records.
        self.communication_ranges = {f"agent_{i+1}": 25 for i in range(agent_count)}
        self.communication_graph = {}
        self.trajectories = {agent: [] for agent in self.possible_agents}

        # For reward shaping.
        self.stacked_values = {agent: 0 for agent in self.possible_agents}
        self.seen={agent: None for agent in self.positions.keys()}      # updated by observe() to indicate if unvisited cells exist in local patch.
        self.nearest_tile = None  # stores location of nearest unvisited tile (when needed).
        self.stop = 300000  # simulation stop step.

        # ----- Mode Scheduling (Exploration vs Relay) -----
        self.mode = 1  # 1: exploration, 0: relay.
        self.mode_start_step = 0
        self.explore = 10000
        self.relay = 10000
        self.exploration_period = self.explore
        self.relay_period = self.relay
        self.score = 0
        self.resources={agent: 0 for agent in self.possible_agents}

        # Discrete-to-continuous action mapping.
        self.disc_action_2_continuous = {
            3: [0, 1],
            7: [0, -1],
            5: [-1, 0],
            0: [1, 0],
            1: [1, 1],
            4: [-1, 1],
            6: [-1, -1],
            2: [1, -1]
        }

        self.energy_consumption_record = {
            agent: {
                'operating_cost': 0.0,
                'data_relay_cost': 0.0,
                'movement_cost': 0.0,
                'merge_map_cost': 0.0,
                'lidar_working_cost': 0.0,
                'computation_cost': 0.0
            }
            for agent in self.possible_agents
        }
        print(f"Obstacle Proportion: {np.sum(self.obstacles == self.omv) / np.prod(self.obstacles.shape):.2f}")
        self.render_train()

    def merge_agent_maps(self):
        """
        For each pair of agents in communication range, merge their personal maps using elementwise minimum.
        Also, update the global map as the union (minimum) of all agent maps.
        """
        for i, agent_a in enumerate(self.possible_agents):
            for j in range(i+1, len(self.possible_agents)):
                agent_b = self.possible_agents[j]
                dist = np.linalg.norm(self.positions[agent_a] - self.positions[agent_b])
                comm_range = min(self.communication_ranges[agent_a], self.communication_ranges[agent_b])
                if dist <= comm_range:
                    merge_cost = 0.001
                    self.power_levels[agent_a] -= merge_cost
                    self.power_levels[agent_b] -= merge_cost
                    self.energy_consumption_record[agent_a]['merge_map_cost'] += merge_cost
                    self.energy_consumption_record[agent_b]['merge_map_cost'] += merge_cost
                    merged_map = np.minimum(self.agent_maps[agent_a], self.agent_maps[agent_b])
                    self.agent_maps[agent_a] = merged_map.copy()
                    self.agent_maps[agent_b] = merged_map.copy()
        # Update global map as the union of all agent maps.
        self.global_map = np.minimum.reduce(list(self.agent_maps.values()))

    def update_mode(self):
        """
        Check whether to switch modes:
        - In exploration mode, switch to relay if the phase lasts longer than the exploration period 
            or no unvisited tile remains in agent_1's map.
        - In relay mode, switch to exploration if all data is delivered or the relay phase times out.
        In each switch, extend the period for the next cycle.
        """
        duration = self.step_count - self.mode_start_step
        if self.mode == 1:
            no_tiles = not np.any(self.agent_maps[self.possible_agents[0]] == self.unvisited_value)
            if duration >= self.exploration_period or no_tiles:
                self.mode = 0
                self.mode_start_step = self.step_count
                self.exploration_period += 200
                # print(f"Switched to Relay mode at step {self.step_count}")
                # Call render when first exploration phase ends
                # if self.step_count < self.exploration_period + 200:
                #     self.render()
        # elif self.mode == 0:
            all_sent = all(val == 0 for val in self.data_collected.values())
            if all_sent or duration >= self.relay_period:
                self.mode = 1
                self.mode_start_step = self.step_count
                self.relay_period += 200
                # print(f"Switched to Exploration mode at step {self.step_count}")
                # Call render when first relay phase ends
                # if self.step_count < self.exploration_period + self.relay_period + 400:
                #     self.render()

    def update_communication_graph(self):
        """
        Build a communication graph: add an edge if agents are within each other's communication range.
        """
        self.communication_graph = {agent: [] for agent in self.positions.keys()}
        for agent1, pos1 in self.positions.items():
            for agent2, pos2 in self.positions.items():
                if agent1 != agent2:
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance <= self.communication_ranges[agent1]:
                        self.communication_graph[agent1].append(agent2)
                    if distance <= self.communication_ranges[agent2]:
                        self.communication_graph[agent2].append(agent1)

    def bfs_find_path(self, start, goal):
        """
        Use breadth-first search to find a communication path (sequence of agents) from start to goal.
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


    
    def step(self, actions):
        """
        Execute one step.
        Behavior depends on current mode:
         - Mode 1 (Exploration): Agents move, explore a local 10×10 patch (updating their personal and global maps), 
           incur movement and sensing power costs, and may deliver data at the base.
         - Mode 0 (Relay): Agents still move and explore but focus on relaying collected data toward the base.
        Rewards (and additional penalties or bonuses) are calculated based on exploration success and power use.
        """
        rewards = {agent: 0.0 for agent in self.positions.keys()}
        infos = {agent: {} for agent in self.positions.keys()}

        # Record trajectories.
        for agent, pos in self.positions.items():
            self.trajectories[agent].append(pos.copy())
            self.power_levels[agent] -=  0.00001647
            self.energy_consumption_record[agent]['computation_cost'] +=  0.00001647
        self.step_count += 1
        base_station = np.array([self.entrance_x, self.entrance_y], dtype=np.float32)
        
        if self.mode == 1:
            # --- Exploration Mode ---
            for agent, action in actions.items():
                # Check available power.
                if self.power_levels[agent] <= 0:
                    self.dones[agent] = True
                    continue

                # Movement cost.
                operating_cost = 0.01
                self.power_levels[agent] -= operating_cost
                self.energy_consumption_record[agent]['operating_cost'] += operating_cost
                self.power_levels[agent] = max(0, self.power_levels[agent])

                # Update observation to set self.seen.
                self.observe(agent)

                # If no unvisited tile is seen in local patch, update nearest target.
                if not self.seen[agent]:
                    self.get_nearest_uncleaned(agent)
                    
                    old_distance = np.linalg.norm(self.nearest_tile - self.positions[agent])
                else:
                    old_distance = None

                # Compute new candidate position.
                action_cont = self.disc_action_2_continuous[int(action[0])]
                raw_position = self.positions[agent] + np.clip(action_cont, -1, 1) * self.timestep
                new_position = np.clip(raw_position, 0, [self.cave_size[0] - 1, self.cave_size[1] - 1])
                # Check for collision against obstacles.
                if (not np.array_equal(raw_position, new_position) or 
                    self.global_map[int(new_position[0]), int(new_position[1])] == self.omv):
                    rewards[agent] -= 2  # penalty for collision/hitting wall
                elif self.global_map[int(new_position[0]), int(new_position[1])] == self.res:
                    if action[1]==0 and agent=='agent_2':  #agent_2 is the excavator 0 for pick 1 for drop
                        self.resources[agent] += 1
                        self.global_map[int(new_position[0]), int(new_position[1])] = -1        
            
                else:
                    self.positions[agent] = new_position
                    if agent == 'agent_2' and action[1] == 1 or agent == 'agent_3' and action[1] == 0:  # agent_2 performing drop action
        
                        distance_2_to_3 = np.linalg.norm(self.positions['agent_2'] - self.positions['agent_3'])
                        
                        if distance_2_to_3 <= 1:  # If they're close enough
                            # Transfer all resources from agent_2 to agent_3
                            resources_to_transfer = self.resources['agent_2']
                            if resources_to_transfer > 0:  # Only transfer if agent_2 has resources
                                self.resources['agent_3'] += resources_to_transfer
                                self.resources['agent_2'] = 0
            
                    move_cost = np.linalg.norm(action_cont) * self.timestep * 0.05
                    self.power_levels[agent] -= move_cost
                    self.energy_consumption_record[agent]['movement_cost'] += move_cost
                    self.power_levels[agent] = max(0, self.power_levels[agent])
                    if self.power_levels[agent] <= 0:
                        continue

                # Exploration update over a 10x10 region.
                new_resources=0
                grid_x, grid_y = int(self.positions[agent][0]), int(self.positions[agent][1])
                new_explorations = 0
                for i in range(grid_x - self.half, grid_x + self.half):
                    for j in range(grid_y - self.half, grid_y + self.half):
                        if 0 <= i < self.cave_size[0] and 0 <= j < self.cave_size[1]:
                            if self.agent_maps[agent][i, j] == self.unvisited_value:
                                self.agent_maps[agent][i, j] = -1
                            if self.global_map[i, j] == self.res:
                                new_resources += 1
                                self.agent_maps[agent][i, j] = self.res
                                # Also update the global map.
                            if self.global_map[i, j] == self.unvisited_value:
                                self.global_map[i, j] = -1
                                new_explorations += 1
                # Data collection and power cost for exploration sensing.
                if new_explorations == 0:
                    rewards[agent] -= 1
                else:
                    lidar_cost = 0.025
                    self.power_levels[agent] -= lidar_cost
                    self.data_collected[agent] += new_explorations
                    self.energy_consumption_record[agent]['lidar_working_cost'] += lidar_cost


                # Deliver data if near base station.
                if (np.linalg.norm(self.positions[agent] - base_station) <= self.communication_ranges[agent] and 
                    self.data_collected[agent] > 0):
                    self.base_info += self.data_collected[agent]
                    data_delivery_cost = self.data_collected[agent] * 0.01
                    self.power_levels[agent] -= data_delivery_cost
                    self.energy_consumption_record[agent]['data_relay_cost'] += data_delivery_cost
                    self.data_collected[agent] = 0
                if (np.linalg.norm(self.positions[agent] - base_station <=0.5)) and agent=='agent_3' and action[1]==1:
                    self.base_resources = self.resources[agent]
                    self.resources[agent] =0

        elif self.mode == 0:
            # --- Relay Mode ---
            for agent, action in actions.items():
                # Check power.
                
                if self.power_levels[agent] <= 0:
                    self.dones[agent] = True
                    continue

                operating_cost = 0.01
                self.power_levels[agent] -= operating_cost
                self.energy_consumption_record[agent]['operating_cost'] += operating_cost
                
                self.power_levels[agent] = max(0, self.power_levels[agent])
                if self.data_collected[agent] ==0 :
                    continue
                self.observe(agent)
                old_distance = np.linalg.norm(base_station - self.positions[agent])
                action_cont = self.disc_action_2_continuous[int(action)]
                raw_position = self.positions[agent] + np.clip(action_cont, -1, 1) * self.timestep
                new_position = np.clip(raw_position, 0, [self.cave_size[0]-1, self.cave_size[1]-1])
                if (not np.array_equal(raw_position, new_position) or 
                    self.obstacles[int(new_position[0]), int(new_position[1])] == self.omv):
                    rewards[agent] -= 2
                    
                elif self.global_map[int(new_position[0]), int(new_position[1])] == self.res:
                    if action[1]==0 and agent=='agent_2':  #agent_2 is the excavator 0 for pick 1 for drop
                        self.resources[agent] += 1
                        self.global_map[int(new_position[0]), int(new_position[1])] = -1
                else:
                    self.positions[agent] = new_position
                    self.power_levels[agent] -= np.linalg.norm(action_cont) * self.timestep * 0.05
                    self.energy_consumption_record[agent]['movement_cost'] += np.linalg.norm(action_cont) * self.timestep * 0.05
                    self.power_levels[agent] = max(0, self.power_levels[agent])
                    if self.power_levels[agent] == 0:
                        continue

                grid_x, grid_y = int(self.positions[agent][0]), int(self.positions[agent][1])
                new_explorations = 0
                new_resources = 0
                for i in range(grid_x - self.half, grid_x + self.half):
                    for j in range(grid_y - self.half, grid_y + self.half):
                        if 0 <= i < self.cave_size[0] and 0 <= j < self.cave_size[1]:
                            if self.agent_maps[agent][i, j] == self.unvisited_value:
                                self.agent_maps[agent][i, j] = -1
                            if self.global_map[i, j] == self.res:
                                new_resources += 1
                                self.agent_maps[agent][i, j] = self.res
                            if self.global_map[i, j] == self.unvisited_value:
                                    self.global_map[i, j] = -1
                                    new_explorations += 1

                if new_explorations > 0:
                    self.power_levels[agent] -= 0.025
                    self.energy_consumption_record[agent]['lidar_working_cost'] += 0.025
                    self.data_collected[agent] += new_explorations

                # Reward for approaching the base station.
                new_distance = np.linalg.norm(base_station - self.positions[agent])
                if new_distance < old_distance and self.data_collected[agent] > 0:
                    rewards[agent] += 0.5
                else:
                    rewards[agent] -= 0.05

            # Deliver data for agents near the base.
            for agent in self.possible_agents:
                if (np.linalg.norm(self.positions[agent] - base_station) <= self.communication_ranges[agent] and 
                    self.data_collected[agent] > 0):
                    rewards[agent] += self.data_collected[agent] * 0.03
                    self.base_info += self.data_collected[agent]
                    data_delivery_cost = self.data_collected[agent] * 0.01
                    self.power_levels[agent] -= data_delivery_cost
                    self.energy_consumption_record[agent]['data_relay_cost'] += data_delivery_cost
                    self.data_collected[agent] = 0

            # Relay data among agents in the network.
            self.update_communication_graph()
            energy_cost_rate = 0.01
            relay_reward_rate = 0.001
            for agent in self.possible_agents:
                if self.data_collected[agent] > 0:
                    agent_dist = np.linalg.norm(self.positions[agent] - base_station)
                    candidates = []
                    for other_agent in self.possible_agents:
                        if other_agent != agent:
                            other_dist = np.linalg.norm(self.positions[other_agent] - base_station)
                            if other_dist < agent_dist:
                                path = self.bfs_find_path(agent, other_agent)
                                if path:
                                    candidates.append((other_agent, path))
                    if candidates:
                        candidate, path = min(candidates,
                                              key=lambda x: (np.linalg.norm(self.positions[agent] - self.positions[x[0]]),
                                                             len(x[1])))
                        for i in range(len(path) - 1):
                            transmitter = path[i]
                            receiver = path[i + 1]
                            info_to_send = self.data_collected[transmitter]
                            if info_to_send > 0:
                                max_transfer_tx = self.power_levels[transmitter] / energy_cost_rate
                                max_transfer_rx = self.power_levels[receiver] / energy_cost_rate
                                transferable = min(info_to_send, max_transfer_tx, max_transfer_rx)
                                if transferable <= 0:
                                    transferable = 0
                                self.power_levels[transmitter] -= transferable * energy_cost_rate
                                self.power_levels[receiver] -= transferable * energy_cost_rate
                                self.energy_consumption_record[transmitter]['data_relay_cost'] += transferable * energy_cost_rate
                                self.energy_consumption_record[receiver]['data_relay_cost'] += transferable * energy_cost_rate
                                rewards[transmitter] += relay_reward_rate * transferable
                                rewards[receiver] += relay_reward_rate * transferable
                                self.data_collected[transmitter] -= transferable
                                self.data_collected[receiver] += transferable

            # Final check again for base station delivery.
            for agent in self.possible_agents:
                if (np.linalg.norm(self.positions[agent] - base_station) <= self.communication_ranges[agent] and 
                    self.data_collected[agent] > 0):
                    rewards[agent] += self.data_collected[agent] * 0.03
                    self.base_info += self.data_collected[agent]
                    data_delivery_cost = self.data_collected[agent] * 0.01
                    self.power_levels[agent] -= data_delivery_cost
                    self.energy_consumption_record[agent]['data_relay_cost'] += data_delivery_cost
                    self.data_collected[agent] = 0

        # Check for simulation stop conditions.
        if self.step_count >= self.stop:
            self.dones = {agent: True for agent in self.possible_agents}
            self.plot_interval_trajectories(0, self.step_count)

        total_cells = np.prod(self.cave_size)
        num_obstacles = np.sum(self.obstacles == self.omv)
        total_non_obstacle = total_cells - num_obstacles

        if all(v == 0 for v in self.data_collected.values()) and self.base_info >= total_non_obstacle:
            self.dones = {agent: True for agent in self.possible_agents}
        # if self.calculate_collectedinfo_proportion() >= 0.9:
        #     self.dones = {agent: True for agent in self.possible_agents}
            
        if all(self.dones.values()):
            self.score = 0
            for agent in self.possible_agents:
                self.score+=( 10*self.calculate_collectedinfo_proportion())**3*(self.power_levels[agent]/self.power+0.5)
                rewards[agent] +=( 5*1)**3*self.power_levels[agent]/self.power
            print(f"Total Score: {self.score:.2f}",end=" ")  
            print(f"Step {self.step_count}: InfoPro: {self.calculate_collectedinfo_proportion():.2f}") 
            self.render()
            self.plot_energy_consumption("energy_consumption.png")
        self.merge_agent_maps()
        self.update_mode()

        obs = {agent: self.observe(agent) for agent in self.possible_agents}
        return obs, rewards, self.dones, self.truncs, infos

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        Reinitialize positions, power levels, maps (global and per-agent), data collected, mode, etc.
        """
        # print(f"obstacle Proportion: {self.calculate_obstacle_proportion():.2f}")
        # print(f"Step {self.step_count}: Exploration Proportion: {self.calculate_exploration_proportion():.2f}")
        # print(f"Info Collected Proportion: {self.calculate_collectedinfo_proportion():.2f}")
        # print(f"Total Score: {self.score:.2f}")
        np.random.seed(30)
        self.step_count = 0
        self.ospro = 0.1
        self.cave_size = (100, 120)
        self.entrance_x = 0
        self.entrance_y = self.cave_size[1] // 2
        self.positions = {
            f"agent_{i+1}": np.array([self.entrance_x + i, self.entrance_y], dtype=np.float32)
            for i in range(self.agent_count)
        }
        self.obstacles = np.random.choice(
            [self.omv, 0,self.res],
            size=(int(self.cave_size[0]), int(self.cave_size[1])),
            p=[self.ospro, 1 - self.ospro-self.res, self.res]
        )
        for value in self.positions.values():
            x, y = int(value[0]), int(value[1])
            self.obstacles[x, y] = self.UNEXPLORED
        self.global_map = np.copy(self.obstacles)

        self.agini_map = np.copy(self.obstacles)
        for i in range(int(self.cave_size[0])):
            for j in range(int(self.cave_size[1])):
                if self.obstacles[i,j]==self.res:
                    self.agini_map[i,j] = self.unvisited_value

        self.score = 0
        self.power_levels = {f"agent_{i+1}": self.power for i in range(self.agent_count)}
        self.data_collected = {f"agent_{i+1}": 0 for i in range(self.agent_count)}
        self.dones = {agent: False for agent in self.positions.keys()}
        self.truncs = {agent: False for agent in self.positions.keys()}
        self.trajectories = {agent: [] for agent in self.possible_agents}
        self.stacked_values = {agent: 0 for agent in self.possible_agents}
        self.nearest_tile = None
        # Reset each agent's personal map.
        self.agent_maps = {agent: np.copy(self.agini_map) for agent in self.possible_agents}
        self.base_info = 0
        # Reset mode scheduling.
        self.mode = 1
        self.mode_start_step = 0
        self.exploration_period = self.explore
        self.relay_period = self.relay
        self.seen={agent: None for agent in self.positions.keys()}
        self.resources={agent: 0 for agent in self.possible_agents}

        self.energy_consumption_record = {
            agent: {
                'operating_cost': 0.0,
                'data_relay_cost': 0.0,
                'movement_cost': 0.0,
                'merge_map_cost': 0.0,
                'lidar_working_cost': 0.0,
                'computation_cost': 0.0
            }
            for agent in self.possible_agents
        }
        return {agent: self.observe(agent) for agent in self.possible_agents}, {}

    def observe(self, agent):
        """
        Return an observation for the agent.
        This consists of a flattened 10×10 patch (from the agent's personal map) plus 2 values:
         - In exploration mode, the vector to the nearest unvisited tile.
         - In relay mode, the vector from the agent to the base station.
        """
        x, y = map(int, self.positions[agent])
        start_x = x - self.half
        end_x = x + self.half
        start_y = y - self.half
        end_y = y + self.half

        neighbor_values = []
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if 0 <= i < self.cave_size[0] and 0 <= j < self.cave_size[1]:
                    neighbor_values.append(self.agent_maps[agent][i, j])
                else:
                    neighbor_values.append(self.omv)
        self.seen[agent] = any(val == self.unvisited_value for val in neighbor_values)
        if self.mode == 1:
            rel_location = self.get_nearest_uncleaned(agent)
        else:
            base_station = np.array([self.entrance_x, self.entrance_y], dtype=np.float32)
            rel_location = (base_station - self.positions[agent]).astype(np.float32)
        obs = np.array(neighbor_values + list(rel_location), dtype=np.float32)
        obs = obs.reshape(self.shape, )
        return obs

    def get_nearest_uncleaned(self, agent):
        """
        Detect the nearest unvisited cell (value == self.unvisited_value) in the agent's personal map.
        Returns the relative vector (dx, dy) from the agent's current position.
        Also saves the absolute coordinate of the nearest unvisited tile.
        Now includes computational cost based on execution time.
        """
        # Check if agent has enough power
        if self.power_levels[agent] <= 0:
            return np.array([0, 0], dtype=np.float32)
        
        # Start measuring time for computation
        import time
        start_time = time.time()
        
        pos = self.positions[agent]
        indices = np.argwhere(self.agent_maps[agent] == self.unvisited_value)
        
        if indices.size == 0:
            # Calculate computation cost even when no unexplored cells found
            end_time = time.time()
            computation_time = end_time - start_time
            search_cost = computation_time * 0.01
            
            self.power_levels[agent] -= search_cost
            self.energy_consumption_record[agent]['computation_cost'] += search_cost
            
            return np.array([0, 0], dtype=np.float32)
        
        diffs = indices - pos
        squared_distances = np.sum(diffs**2, axis=1)
        min_idx = np.argmin(squared_distances)
        self.nearest_tile = indices[min_idx]
        
        # Calculate computation cost based on time taken
        end_time = time.time()
        computation_time = end_time - start_time
        search_cost = computation_time * 0.01
        
        self.power_levels[agent] -= search_cost
        self.energy_consumption_record[agent]['computation_cost'] += search_cost
        
        return (self.nearest_tile - pos).astype(np.float32)

    def state(self):
        return {a: self.observe(a) for a in self.possible_agents}

    def available_actions(self, agent):
        return None

    def calculate_obstacle_proportion(self):
        return np.sum(self.obstacles == self.omv) / np.prod(self.obstacles.shape)

    def calculate_exploration_proportion(self):
        visited = np.sum(self.global_map == -1)
        total_cells = np.prod(self.cave_size)
        num_obstacles = np.sum(self.obstacles == self.omv)
        total_non_obstacle = total_cells - num_obstacles
        return visited / total_non_obstacle

    def calculate_collectedinfo_proportion(self):
        total_cells = np.prod(self.cave_size)
        num_obstacles = np.sum(self.obstacles == self.omv)
        total_non_obstacle = total_cells - num_obstacles
        return self.base_info / total_non_obstacle

    def plot_interval_trajectories(self, start_step, end_step):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.global_map.T, origin='lower', cmap='gray')
        for agent, positions in self.trajectories.items():
            interval_positions = np.array(positions[start_step:end_step])
            plt.plot(interval_positions[:, 0], interval_positions[:, 1],
                     linestyle='-', linewidth=0.8, label=agent)
        plt.title(
            f"Mode: {self.mode} | Obstacles Proportion: {self.calculate_obstacle_proportion():.2f} | Steps: {self.step_count}\n"
            f"Exploration Proportion: {self.calculate_exploration_proportion():.2f}, "
            f"Info Collected Proportion: {self.calculate_collectedinfo_proportion():.2f}"
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.savefig("Plot")
        plt.close()

    def render(self):
        plt.figure(figsize=(12, 6))
        plt.imshow(self.global_map.T, origin='lower', cmap='gray')
        
        # Draw lidar coverage bounding boxes for each agent
        for agent, pos in self.positions.items():
            x, y = pos
            # Create a rectangle patch for the lidar coverage (10x10 area)
            # half-width of the lidar coverage area is self.half (which is 5)
            rect = plt.Rectangle(
                (x - self.half, y - self.half),  # bottom-left corner
                2 * self.half,  # width
                2 * self.half,  # height
                linewidth=1,
                edgecolor='yellow',
                facecolor='none',
                alpha=0.7,
                linestyle='--',
                label=f"{agent} Lidar" if agent == self.possible_agents[0] else ""
            )
            plt.gca().add_patch(rect)
        
        # Plot agent trajectories
        distances = {
            agent: np.linalg.norm(self.positions[agent] - np.array([self.entrance_x, self.entrance_y]))
            for agent in self.positions.keys()
        }
        relaying_agent = min(distances, key=distances.get)
        
        for agent, positions in self.trajectories.items():
            if len(positions) > 1:
                trajectory = np.array(positions)
                if agent == relaying_agent:
                    plt.plot(trajectory[:, 0], trajectory[:, 1],
                            linestyle='--', linewidth=0.8, label=f"{agent}")
                else:
                    plt.plot(trajectory[:, 0], trajectory[:, 1],
                            linestyle='--', linewidth=0.8, label=f"{agent}")
                            
        # Add current agent positions with different markers
        for agent, pos in self.positions.items():
            plt.scatter(pos[0], pos[1], marker='o', s=100, 
                    color='red' if agent == relaying_agent else 'green',
                    label=f"{agent} position" if agent == self.possible_agents[0] else "")
        
        plt.scatter(self.entrance_x, self.entrance_y, marker='*', color='blue',
                    s=100, label="Base Station")
                    
        plt.title(
            f"Mode: {self.mode} | Obstacles Proportion: {self.calculate_obstacle_proportion():.2f} | Steps: {self.step_count}\n"
            f"Exploration Proportion: {self.calculate_exploration_proportion():.2f}, "
            f"Info Collected Proportion: {self.calculate_collectedinfo_proportion():.2f}"
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        
        # Create a custom legend to avoid duplicate entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.grid()
        plt.show()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render_train(self):
        plt.imshow(self.global_map, cmap="gray")
        plt.title("Cave Map")
        plt.show()
    
    def plot_energy_consumption(self, filename=None):
        """
        Plot a horizontal stacked bar chart showing energy consumption by category
        for each agent. Similar to the first code's plot_energy_consumption function.
        Shows the energy consumption value directly on each bar segment except for
        merge map and computation categories.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        agents = self.possible_agents
        operating = [self.energy_consumption_record[a]['operating_cost'] for a in agents]
        relay     = [self.energy_consumption_record[a]['data_relay_cost'] for a in agents]
        move      = [self.energy_consumption_record[a]['movement_cost'] for a in agents]
        merge     = [self.energy_consumption_record[a]['merge_map_cost'] for a in agents]
        lidar     = [self.energy_consumption_record[a]['lidar_working_cost'] for a in agents]
        a_star    = [self.energy_consumption_record[a]['computation_cost'] for a in agents]  # New category
        
        # Total consumed (sum of all cost categories)
        total_consumed = []
        for i in range(len(agents)):
            agent_consumed = operating[i] + relay[i] + move[i] + merge[i] + lidar[i] + a_star[i]
            total_consumed.append(agent_consumed)
            
        remain = [self.power_levels[a] for a in agents]
        remain_ratio = [r / self.power for r in remain]
        y_pos = np.arange(len(agents))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Create the bars
        left_vals = np.zeros(len(agents))
        
        # Function to add value labels centered in each bar segment
        # Modified to accept a list of categories to skip labeling
        def add_labels(rects, values, skip_labels=False, color='white'):
            if skip_labels:
                return
                
            for i, (rect, value) in enumerate(zip(rects, values)):
                width = rect.get_width()
                if width > 0:  # Only label non-zero values
                    # Calculate center position for text
                    x = rect.get_x() + width/2
                    y = rect.get_y() + rect.get_height()/2
                    
                    # Display value on the bar
                    label = f'{value:.1f}'
                    ax.text(x, y, label, ha='center', va='center', 
                            color=color, fontweight='bold', fontsize=9)
        
        # Operating cost
        p1 = ax.barh(y_pos, operating, color='purple', left=left_vals, label='Operating')
        add_labels(p1, operating)
        left_vals += operating
        
        # Data relay cost
        p2 = ax.barh(y_pos, relay, color='green', left=left_vals, label='Data Relay')
        add_labels(p2, relay)
        left_vals += relay
        
        # Movement cost
        p3 = ax.barh(y_pos, move, color='blue', left=left_vals, label='Movement')
        add_labels(p3, move)
        left_vals += move
        
        # Merge map cost - skip labels
        p4 = ax.barh(y_pos, merge, color='orange', left=left_vals, label='Merge Map')
        add_labels(p4, merge, skip_labels=True)
        left_vals += merge
        
        # Lidar cost
        p5 = ax.barh(y_pos, lidar, color='brown', left=left_vals, label='LIDAR')
        add_labels(p5, lidar)
        left_vals += lidar
        
        # A* computation cost - skip labels
        p6 = ax.barh(y_pos, a_star, color='red', left=left_vals, label='Computation')
        add_labels(p6, a_star, skip_labels=True)
        left_vals += a_star

        ax.set_yticks(y_pos)
        ax.set_yticklabels(agents)
        
        # Mark the agents with special roles (if applicable)
        # In this code, first agent might be special since it relays to base
        try:
            if len(agents) > 0:
                ax.get_yticklabels()[0].set_color('red')
                ax.get_yticklabels()[0].set_fontweight('bold')
        except:
            pass
        
        ax.set_xlabel("Energy Consumption (absolute units)")
        ax.set_title("Energy Consumption by Category & Remaining Power Ratio")
        ax.legend(loc='best')
        
        # Annotate each bar with total used energy and remaining percentage
        for i, agent in enumerate(agents):
            highlight = ""
            text_color = 'black'
            font_weight = 'normal'
            
            # Highlight agent_1 as it has special role in data relay to base
            if agent == "agent_1":
                highlight = "★ BASE RELAY AGENT ★ "
                text_color = 'red'
                font_weight = 'bold'
                
            ax.text(
                left_vals[i] + 0.05,
                i,
                f"{highlight}Used={left_vals[i]:.2f}, Remain={remain_ratio[i]*100:.1f}%",
                va='center',
                fontsize=8,
                fontweight=font_weight,
                color=text_color
            )
        
        plt.tight_layout()
        if filename:
           plt.savefig(filename)
        plt.show()
