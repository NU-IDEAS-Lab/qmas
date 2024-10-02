from pettingzoo.utils.env import ParallelEnv
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

from flocking_zoo.env.communication_model import CommunicationModel
from flocking_zoo.env.flocking_graph import FlockingGraph
from flocking_zoo.env.entity import Shepherd, Sheep, ENTITY_TYPE

    
class parallel_env(ParallelEnv):
    metadata = {
        "name": "flocking_zoo_environment_v0",
    }

    def __init__(self,
                 num_shepherds = 3,
                 num_sheep = 5,
                 comms_model = CommunicationModel(model = "none"),
                 speed_shepherds = 1.0,
                 speed_sheep = 1.0,
                 sheep_force_avoid_sheep = 10.0,
                 sheep_force_avoid_shepherd = 100.0,
                 sheep_force_align = 1.0,
                 sheep_force_cluster = 1.0,
                 sheep_radius_avoid_sheep = np.inf,
                 sheep_radius_avoid_shepherd = np.inf,
                 alpha = 10.0,
                 beta = 100.0,
                 action_method = "velocity",
                 reward_method_terminal = "time",
                 observation_radius = np.inf,
                 observe_method = "graph",
                 observe_method_global = None,
                 attrition_method = "none",
                 attrition_random_probability = 0.0,
                 attrition_min_agents = 2,
                 attrition_times = [],
                 world_dimensions = (100, 100),
                 max_cycles: int = -1,
                 max_nodes: int = 50,
                 reward_interval: int = -1,
                 randomize_graph: bool = False,
                 regenerate_graph_on_reset: bool = True,
                 random_graph_danger_zones: int = 3,
                 random_graph_grid_size: int = 20,
                 random_graph_grid_border: int = 10,
                 *args,
                 **kwargs):
        """
        Initialize the PatrolEnv object.

        Args:
            patrol_graph (PatrolGraph): The patrol graph representing the environment.
            num_agents (int): The number of agents in the environment.

        Returns:
            None
        """
        super().__init__()


        # Configuration.
        self.observationRadius = observation_radius
        self.max_cycles = max_cycles
        self.comms_model = comms_model
        self.sheep_force_avoid_sheep = sheep_force_avoid_sheep
        self.sheep_force_avoid_shepherd = sheep_force_avoid_shepherd
        self.sheep_force_align = sheep_force_align
        self.sheep_force_cluster = sheep_force_cluster
        self.sheep_radius_avoid_sheep = sheep_radius_avoid_sheep
        self.sheep_radius_avoid_shepherd = sheep_radius_avoid_shepherd
        self.action_method = action_method
        self.reward_method_terminal = reward_method_terminal
        self.observe_method = observe_method
        self.observe_method_global = observe_method_global if observe_method_global != None else observe_method
        self.attrition_method = attrition_method
        self.attrition_random_probability = attrition_random_probability
        self.attrition_times = attrition_times
        self.attrition_min_agents = attrition_min_agents
        self.max_nodes = max_nodes
        self.world_dimensions = np.array(world_dimensions)
        self.reward_interval = reward_interval
        self.alpha = alpha
        self.beta = beta
        self.randomize_graph = randomize_graph
        self.regenerate_graph_on_reset = regenerate_graph_on_reset
        
        self.num_danger_zones = random_graph_danger_zones if randomize_graph else 2 # Default from FlockingGraph.initialize_example()

        # Create the environment graph.
        self.graph = FlockingGraph(
            generate_random_graph = randomize_graph,
            num_danger_zones = random_graph_danger_zones,
            grid_size = random_graph_grid_size,
            grid_border = random_graph_grid_border,
            world_dimensions=self.world_dimensions
        )

        # Create the agents with random starting positions.
        self.possible_agents = [
            Shepherd(self.graph,
                maxSpeed = speed_shepherds,
                observationRadius = self.observationRadius
            ) for i in range(num_shepherds)
        ]
        self.possible_sheep = [
            Sheep(self.graph,
                maxSpeed = speed_sheep,
                observationRadius = self.observationRadius
            ) for i in range(num_sheep)
        ]

        # Create the action space.
        action_space = self._buildActionSpace(self.action_method)

        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        self.state_space = self._buildStateSpace(self.observe_method_global)
        
        # Create the observation space.
        obs_space = self._buildStateSpace(self.observe_method)

        # Set up the lists of per-agent action and observation spaces.
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore
        self.observation_spaces = spaces.Dict({agent: obs_space for agent in self.possible_agents}) # type: ignore
        
        # Set up some values we may want to render
        self.last_reward_dict = None

        self.reset_count = 0
        self.reset()


    def _buildActionSpace(self, action_method):
        ''' Creates a gym.spaces.* object representing the action space. '''

        if action_method == "velocity":
            return spaces.Box(
                low = np.array([-1.0, -1.0]),
                high = np.array([1.0, 1.0]),
                dtype = np.float32
            )
        elif action_method == "angle_speed":
            return spaces.Box(
                # angle (radians), speed (fraction of max)
                low = np.array([0.0, 0.0]),
                high = np.array([2 * math.pi, 1.0]),
                dtype = np.float32
            )
        elif action_method == "position":
            return spaces.Box(
                low = np.array([-np.inf, -np.inf]),
                high = np.array([np.inf, np.inf]),
                dtype = np.float32
            )
        else:
            raise ValueError(f"Invalid action method {action_method}")


    def _buildStateSpace(self, observe_method):
        ''' Creates a state space given the observation method.
            Returns a gym.spaces.* object. '''
        
        # Create the state space dictionary.
        state_space = {}

        # Add to the dictionary depending on the observation method.
        if observe_method in ["graph", "graph_angles", "matrix"]:
        # if observe_method in ["graph", "graph_angles"]:        
            state_space["id"] = spaces.Box(
                low=np.array([0.0]),
                high=np.array([np.inf]),
                dtype=np.float32
            )
        
        if observe_method in ["matrix"]:
            state_space["matrix"] = spaces.Dict({i: spaces.Box(
                low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                dtype=np.float32
            ) for i in range(len(self.possible_agents) + len(self.possible_sheep) + self.num_danger_zones + 1)})
            # All shepherds + all sheep + danger zones + goal zone
            # ) for i in range(1)}) # Just send the relative goal zone
        
        if observe_method in ["sheep_distances"]:
            # This observation method is for use by the critic.
            state_space["sheep_distances"] = spaces.Box(
                low = np.array([0.0] * len(self.possible_sheep)),
                high = np.array([np.inf] * len(self.possible_sheep)),
                dtype=np.float32
            )

        if observe_method in ["graph"]:
            edge_space = spaces.Box(
                # weight (distance)
                low = np.array([0.0]),
                high = np.array([np.inf]),
                dtype=np.float32
            )
            node_space = spaces.Box(
                # nodeType, posX, posY, velX, velY
                low = np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf]),
                high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
                dtype=np.float32
            )
            node_type_idx = 0

            state_space["graph"] = spaces.Graph(
                node_space = node_space,
                edge_space = edge_space
            )
            state_space["graph"].node_type_idx = node_type_idx

        if observe_method in ["graph_angles"]:
            edge_space = spaces.Box(
                # weight (distance), theta (angle)
                low = np.array([0.0, -np.inf]),
                high = np.array([np.inf, np.inf]),
                dtype=np.float32
            )
            node_space = spaces.Box(
                # nodeType
                low = np.array([0.0]),
                high = np.array([np.inf]),
                dtype=np.float32
            )
            node_type_idx = 0

            state_space["graph"] = spaces.Graph(
                node_space = node_space,
                edge_space = edge_space
            )
            state_space["graph"].node_type_idx = node_type_idx
                
        # Convert the dictionary to a gymnasium.spaces object.
        if type(state_space) == dict:
            state_space = spaces.Dict(state_space)
        
        return state_space


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        self.reset_count += 1

        if seed != None:
            random.seed(seed)

        # Reset graph.
        regen = self.randomize_graph and self.regenerate_graph_on_reset
        self.graph.reset(regenerateGraph=regen)

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        for agent in self.agents:
            agent.reset(reset_start_position=regen)
        
        # Reset the sheep.
        self.sheep = copy(self.possible_sheep)
        for sheep in self.sheep:
            sheep.reset(reset_start_position=regen)
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)

        # Return the initial observation.
        observation = {}
        for agent in self.agents:
            visible_shepherds, visible_sheep, visible_others = self.get_visible_entities(agent, allow_done_agents=False)
            visible_entities = visible_shepherds + visible_sheep + visible_others
            self.update_beliefs(agent, visible_entities)
            observation[agent] = self.observe(agent, visible_entities)

        info = {
            agent: {
                "ready": True
            } for agent in self.agents
        }

        return observation, info


    def render(self, figsize=(5, 5)):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).
                
            Returns:
                None
        '''
        plt.figure(figsize=figsize)
        
        # Set up axes.
        # plt.xlim(-self.world_dimensions[0] / 2, self.world_dimensions[0] / 2)
        # plt.ylim(-self.world_dimensions[1] / 2, self.world_dimensions[1] / 2)
        plt.xlim(0.0, self.world_dimensions[0])
        plt.ylim(0.0, self.world_dimensions[1])

        # Draw manually.
        entities = self.agents + self.sheep
        for i in entities:
            pos = i.position
            vel = i.velocity / (np.linalg.norm(i.velocity) + 1e-9)
            if i.entity_type == ENTITY_TYPE.SHEPHERD:
                plt.scatter(*pos, color='red', marker='p', zorder=10, alpha=0.3, s=100, label='Shepherd')
            elif i.entity_type == ENTITY_TYPE.SHEEP:
                plt.scatter(*pos, color='blue', marker='p', zorder=10, alpha=0.3, s=100, label='Sheep')
            plt.quiver(*pos, *vel, color='black', scale=10, scale_units='inches', label='Velocity')

        # Draw the goal and danger zones.
        start = plt.Circle(self.graph.start.position, self.graph.start.radius, color='blue', alpha=0.5, label='Start Zone')
        plt.gca().add_artist(start)
        goal = plt.Circle(self.graph.goal.position, self.graph.goal.radius, color='green', alpha=0.5, label='Goal Zone')
        plt.gca().add_artist(goal)
        for danger_zone in self.graph.danger_zones:
            danger = plt.Circle(danger_zone.position, danger_zone.radius, color='red', alpha=0.5, label='Danger Zone')
            plt.gca().add_artist(danger)

        # Create legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left")

        # Add status text.
        plt.gcf().text(0,0,f'Current step: {self.step_count}')
        plt.gcf().text(0.5,0,f'Average reward last step: {np.mean(list(self.last_reward_dict.values())):.5f}')

        plt.show()


    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''
        return self.observation_spaces[agent]


    def action_space(self, agent):
        ''' Returns the action space for the given agent. '''
        return self.action_spaces[agent]


    def state(self):
        ''' Returns the global state of the environment.
            This is useful for centralized training, decentralized execution. '''
        
        visible_entities = self.possible_agents + self.sheep + [self.graph.goal] + self.graph.danger_zones
        return self._populateStateSpace(self.observe_method_global, self.possible_agents[0], visible_entities)


    def state_all(self):
        ''' Similar to the state() method, but this returns a customized copy of the state space for each agent.
            This is useful for centralized training, decentralized execution. '''
        
        state = {}
        visible_entities = self.possible_agents + self.sheep + [self.graph.goal] + self.graph.danger_zones
        for agent in self.possible_agents:
            state[agent] = self._populateStateSpace(self.observe_method_global, agent, visible_entities)
        return state


    def observe(self, agent, visible_entities=None):
        ''' Returns the observation for the given agent.'''

        if visible_entities == None:
            visible_shepherds, visible_sheep, visible_others = self.get_visible_entities(agent, allow_done_agents=False)
            visible_entities = visible_shepherds + visible_sheep + visible_others

        return self._populateStateSpace(self.observe_method, agent, visible_entities)


    def get_visible_entities(self, receiver, allow_done_agents=False, radius=None):
        ''' Simulates messages sent by all agents to receiver.
            Returns a tuple of visible (shepherds, sheep, other) nodes. '''


        # Track the visible entities.
        visible_shepherds = []
        visible_sheep = []
        visible_others = [self.graph.goal] + self.graph.danger_zones

        # Determine which agents can communicate.
        if allow_done_agents:
            agentList = self.possible_agents
        else:
            agentList = self.agents
        
        # Perform communication.
        for sender in agentList:
            if sender == receiver or self.comms_model.canReceive(sender, receiver):
                for v in self.graph.graph.nodes:
                    pos = self.graph.getNodePosition(v)
                    r = sender.observationRadius if radius == None else radius
                    if self._dist(pos, sender.position) <= r:
                        if v.entity_type == ENTITY_TYPE.SHEPHERD:
                            visible_shepherds.append(v)
                        elif v.entity_type == ENTITY_TYPE.SHEEP:
                            visible_sheep.append(v)
        
        return visible_shepherds, visible_sheep, visible_others


    def update_beliefs(self, agent, visible_entities):
        ''' Updates the agent's state beliefs. '''

        entities = self.agents + self.sheep + [self.graph.goal] + self.graph.danger_zones

        # Update beliefs for visible agents.
        for e in entities:
            if e in visible_entities:
                pos = e.position
                vel = e.velocity
                agent.stateBelief[e] = [pos, vel]
            else:
                # If not visible, assume they keep moving with their last velocity
                agent.stateBelief[e][0] += agent.stateBelief[e][1]


    def _populateStateSpace(self, observe_method, agent, visible_entities):
        ''' Returns a populated state/observation space.'''

        # Perform sorting to ensure observations are always in the same order.
        visible_entities = sorted(visible_entities, key=lambda x: x.id)
        
        # Build observation dictionary.
        obs = {}

        if observe_method in ["graph", "graph_angles", "matrix"]:
            obs["id"] = list(self.graph.graph.nodes).index(agent)

        if observe_method in ["matrix"]:
            # include observations as single matrix.
            entities = self.possible_agents + self.possible_sheep + [self.graph.goal] + self.graph.danger_zones
            obs["matrix"] = np.zeros((len(entities), 6))
            for i, node in enumerate(entities):
                pos, vel = agent.stateBelief[node]
                rel_pos = pos - agent.position
                rel_vel = vel - agent.velocity
                obs["matrix"][i] = np.array([
                    i,
                    node.entity_type,
                    rel_pos[0] / self.world_dimensions[0],
                    rel_pos[1] / self.world_dimensions[1],
                    rel_vel[0] / self.world_dimensions[0],
                    rel_vel[1] / self.world_dimensions[1]
                ])

        if observe_method in ["sheep_distances"]:
            weights = np.array([self.graph.graph[sheep][self.graph.goal]["weight"] for sheep in self.sheep])
            weights_normalized = self._minMaxNormalize(weights, minimum=0.0, maximum=self.graph.goal.radius)
            obs["sheep_distances"] = weights_normalized

        if observe_method in ["graph"]:
            edge_attrs = ["weight"]
            node_attrs = ["type", "pos_x", "pos_y", "vel_x", "vel_y"]
        elif observe_method in ["graph_angles"]:
            # Select attributes to pass to the GNN.
            edge_attrs = ["weight", "theta"]
            node_attrs = ["type"]

        if observe_method in ["graph", "graph_angles"]:
            # Create a copy of the graph.
            g = self.graph.graph.copy()

            # Normalize the edge weights of g.
            weights = nx.get_edge_attributes(g, 'weight')
            # maxWeight = max(weights.values())
            # minWeight = min(weights.values())
            # for edge in g.edges:
            #     g.edges[edge]["weight"] = self._minMaxNormalize(weights[edge], minimum=minWeight, maximum=maxWeight)
            
            goal_x, goal_y = self.graph.goal.position
            goal_theta = math.atan2(goal_y - agent.position[1], goal_x - agent.position[0])
            rotation_theta = np.pi/2.0 - goal_theta
            
            # Construct a rotation matrix such that the goal is on the y-axis
            rotation_matrix = np.array(
                [
                    [np.cos(rotation_theta), -np.sin(rotation_theta)],
                    [np.sin(rotation_theta), np.cos(rotation_theta)],
                ]
            )
            
            # # Trim the graph to only include the nodes and edges that are visible to the agent.
            # subgraph = nx.subgraph(g, visible_entities)

            # Replace the node attributes with the agent's beliefs.
            for node in g.nodes:
                pos, vel = agent.stateBelief[node]
                rel_pos = pos - agent.position
                rel_vel = vel - agent.velocity
                
                rot_pos = rotation_matrix @ rel_pos
                rot_vel = rotation_matrix @ rel_vel
                
                g.nodes[node]["pos_x"] = rot_pos[0] / np.linalg.norm(self.world_dimensions)
                g.nodes[node]["pos_y"] = rot_pos[1] / np.linalg.norm(self.world_dimensions)
                g.nodes[node]["vel_x"] = rot_vel[0] / np.linalg.norm(self.world_dimensions)
                g.nodes[node]["vel_y"] = rot_vel[1] / np.linalg.norm(self.world_dimensions)

            # Convert g to PyG
            data = from_networkx(
                g, #subgraph,
                group_node_attrs=node_attrs,
                group_edge_attrs=edge_attrs
            )
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()

            # Calculate the agent_mask based on the graph node ID assigned to this agent.
            idx = list(g.nodes).index(agent)
            agent_mask = np.zeros(data.num_nodes, dtype=bool)
            agent_mask[idx] = True
            data.agent_idx = idx
            data.agent_mask = agent_mask

            # Add graph to the observation dictionary.
            obs["graph"] = data


        if (type(obs) == dict and obs == {}) or (type(obs) != dict and len(obs) < 1):
            raise ValueError(f"Invalid observation method {self.observe_method}")
        

        # Check if type of any values in obs is a graph.
        if type(obs) == dict:
            # Ensure dictionary ordering.
            obs = dict(sorted(obs.items()))

            typeSet = set([type(v) for v in obs.values()])
            if Data in typeSet:
                # If so, we want the observation to be a single-element array of objects.
                o = np.empty((len(obs),), dtype=object)
                for i, k in enumerate(obs.keys()):
                    o[i] = obs[k]
                obs = o

        return obs
    

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
            agent.id: {
                "ready": self.dones[agent], #if done, set ready to true. Otherwise we will have buffer size problems in MAPPO due to lack of insertion.
            } for agent in self.possible_agents
        }

        # Perform attrition.
        if len(self.agents) >= self.attrition_min_agents:
            if self.attrition_method == "random":
                if random.random() < self.attrition_random_probability:
                    random_index = random.randrange(len(self.agents))
                    attrition_agent = self.agents.pop(random_index)
                    self.dones[attrition_agent] = True
                    print(f"Agent {attrition_agent.id} has been removed from the environment at step {self.step_count}.")
            elif self.attrition_method == "fixed_time":
                if self.step_count in self.attrition_times:
                    random_index = random.randrange(len(self.agents))
                    attrition_agent = self.agents.pop(random_index)
                    self.dones[attrition_agent] = True
                    print(f"Agent {attrition_agent.id} has been removed from the environment at step {self.step_count}.")

        # Update shepherd positions.
        for agent in self.agents:
            if agent in action_dict:
                action = action_dict[agent]

                if type(agent) != Shepherd:
                    raise ValueError(f"Invalid agent type {type(agent)} for action {action}.")

                # Check whether the action is valid.
                if not self.action_space(agent).contains(action):
                    raise ValueError(f"Invalid action {action} for agent {agent}.")

                # Update agent position.
                if self.action_method == "velocity":
                    # agent.velocity = action
                    # agent.position += agent.velocity * agent.maxSpeed
                    
                    # The agent's action is in the local frame, need to convert to global
                    goal_x, goal_y = self.graph.goal.position
                    goal_theta = math.atan2(goal_y - agent.position[1], goal_x - agent.position[0])
                    rotation_theta = -(np.pi/2.0 - goal_theta) # Inverse of what's in _populateStateSpace
                    
                    rotation_matrix = np.array(
                        [
                            [np.cos(rotation_theta), -np.sin(rotation_theta)],
                            [np.sin(rotation_theta), np.cos(rotation_theta)],
                        ]
                    )
                    
                    rot_action = rotation_matrix @ action
                    agent.velocity = rot_action * agent.maxSpeed
                    agent.position += agent.velocity
                    
                elif self.action_method == "angle_speed":
                    angle = action[0]
                    unitSpeed = np.clip(action[1], 0.0, 1.0)
                    speed = unitSpeed * agent.maxSpeed
                    agent.velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
                    agent.position += agent.velocity
                elif self.action_method == "position":
                    agent.velocity = (action - agent.position) / (np.linalg.norm(action - agent.position) + 1e-9) * agent.maxSpeed
                    agent.position += agent.velocity
                else:
                    raise ValueError(f"Invalid action method {self.action_method}")

                # Action complete. Agent is ready for a new one.
                info_dict[agent.id]["ready"] = True
        
        # Collect positions.
        positionsSheep = np.array([sheep.position for sheep in self.sheep])
        velocitiesSheep = np.array([sheep.velocity for sheep in self.sheep])
        positionsShepherds = np.array([shepherd.position for shepherd in self.agents])

        # Calculate flock centroid.
        flock_centroid_prev = np.mean(positionsSheep, axis=0)

        # Perform sheep dynamics.
        for sheep in self.sheep:
            force = np.array([0.0, 0.0])

            distances_sheep = np.linalg.norm(positionsSheep - sheep.position, axis=1, keepdims=True) + 1e-9
            distances_shepherds = np.linalg.norm(positionsShepherds - sheep.position, axis=1, keepdims=True) + 1e-9
            speeds_sheep = np.linalg.norm(velocitiesSheep, axis=1, keepdims=True) + 1e-9

            radius_mask_sheep = distances_sheep < self.sheep_radius_avoid_sheep
            radius_mask_shepherds = distances_shepherds < self.sheep_radius_avoid_shepherd

            # Any distances greater than radius, set to infinity.
            distances_sheep = np.where(radius_mask_sheep, distances_sheep, np.inf)
            distances_shepherds = np.where(radius_mask_shepherds, distances_shepherds, np.inf)
            
            force += -self.sheep_force_avoid_sheep * np.sum((positionsSheep - sheep.position) / distances_sheep ** 3, axis=0)

            force += self.sheep_force_align / len(self.sheep) * np.sum(velocitiesSheep / speeds_sheep, axis=0)

            force += self.sheep_force_cluster / len(self.sheep) * np.sum((positionsSheep - sheep.position) / distances_sheep, axis=0)

            force += -self.sheep_force_avoid_shepherd * np.sum((positionsShepherds - sheep.position) / distances_shepherds ** 3, axis=0)
            
            sheep.velocity = force / (np.linalg.norm(force) + 1e-9) * sheep.maxSpeed
            sheep.position += sheep.velocity
        
        # Calculate new flock information.
        positionsSheep = np.array([sheep.position for sheep in self.sheep])
        flock_centroid = np.mean(positionsSheep, axis=0)
        sheep_in_goal = self.graph.count_in_goal([sheep.position for sheep in self.sheep])

        # Perform post-movement calculations.
        for agent in self.possible_agents:
            
            # Calculate the visible entities (both communicated and observed).
            visible_shepherds, visible_sheep, visible_others = self.get_visible_entities(agent, allow_done_agents=False)
            visible_entities = visible_shepherds + visible_sheep + visible_others

            # Update the agent's state beliefs.
            self.update_beliefs(agent, visible_entities)

            # Perform observation.
            obs_dict[agent] = self.observe(agent, visible_entities)

            # # Give a reward based on improvement in centroid distance to goal.
            # reward_dict[agent] += self.alpha * ((self._dist(flock_centroid_prev, self.graph.goal.position) - self._dist(flock_centroid, self.graph.goal.position)))
            
            # # Give a reward based on centroid distance to goal
            # reward_dict[agent] += 0.01 * (1 / (10 * (self._dist(flock_centroid, self.graph.goal.position) / np.linalg.norm(self.world_dimensions)) + 1))
            
            # Give a reward based on individual sheep distances to goal
            for sheep in self.sheep:
                reward_dict[agent] += (0.01 / len(self.sheep)) * (1.0 / (10.0 * (self._dist(sheep.position, self.graph.goal.position) / np.linalg.norm(self.world_dimensions)) + 1))

            # # Give reward based on number of sheep in the observation.
            # reward_dict[agent] += self.alpha * len(visible_sheep) / len(self.sheep)

            # # Provide a global reward depending on how many sheep are in the goal at
            # # each step in time
            # reward_dict[agent] += self.beta * sheep_in_goal / float(len(self.sheep))
            
            # Provide a small reward for the agent being close to the flock
            # reward_dict[agent] += 0.001 * (1 / (10 * (self._dist(agent.position, flock_centroid) / np.linalg.norm(self.world_dimensions)) + 1))
            
            # Reward agent near to goal
            # reward_dict[agent] += 0.01 * (1 / (10 * (self._dist(agent.position, self.graph.goal.position) / np.linalg.norm(self.world_dimensions)) + 1))
            
        
        # Record miscellaneous information.
        info_dict["agent_count"] = len(self.agents)
        
        # Truncate if all sheep are in the goal region.
        # if sheep_in_goal >= len(self.sheep):
        #     lastStep = True

        # Check truncation conditions.
        if lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles):
            for agent in self.agents:
                # Provide an end-of-episode reward.
                # if self.reward_method_terminal == "time":
                #     reward_dict[agent] -= self.beta * self.step_count
                # elif self.reward_method_terminal == "sheep_in_goal":
                #     # We provide the sheep in goal reward again.
                #     reward_dict[agent] += self.beta * sheep_in_goal / float(len(self.sheep))
                # elif self.reward_method_terminal != "none":
                #     raise ValueError(f"Invalid terminal reward method {self.reward_method_terminal}")

                info_dict[agent.id]["ready"] = True
            
                truncated_dict[agent] = True
            self.agents = []

        # Provide a step penalty to encourage timely completion.
        # for agent in self.agents:
        #         reward_dict[agent] -= 0.01

        # Record some data.
        self.last_reward_dict = reward_dict

        return obs_dict, reward_dict, self.dones, truncated_dict, info_dict


    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    

    def _minMaxNormalize(self, x, eps=1e-8, a=0.0, b=1.0, maximum=None, minimum=None):
        ''' Normalizes numpy array x to be between a and b. '''

        if maximum is None:
            maximum = np.max(x)
        if minimum is None:
            minimum = np.min(x)
        return a + (x - minimum) * (b - a) / (maximum - minimum + eps)