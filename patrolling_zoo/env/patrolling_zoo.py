from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import os
import patrolling_zoo.graphs
from patrolling_zoo.env.communication_model import CommunicationModel
from patrolling_zoo.env.patrol_graph import PatrolGraph, NODE_TYPE
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
from collections import Counter


def add_args(parser):
    ''' Adds environment arguments. '''

    import os.path
    import patrolling_zoo.graphs
    import argparse

    # Define all arguments.
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight of local reward.")
    parser.add_argument("--beta", type=float, default=1000.0,
                        help="Weight of global reward.")
    parser.add_argument("--reward_comms_penalty_weight", type=float, default=1.0,
                        help="Weight of communication reward.")
    parser.add_argument("--graph_file", type=str,
                        default=os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "cumberland.graph"), 
                        help="The path to the graph file.")
    parser.add_argument("--graph_name", type=str,
                        default="cumberland", 
                        help="which graph to run on.")
    parser.add_argument("--graph_random", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to use a random graph.")
    parser.add_argument("--graph_random_nodes", type=int,
                        default=40,
                        help="The number of random nodes to generate.")
    parser.add_argument("--graph_random_radius", type=float,
                        default=75.0,
                        help="The connection radius for random graph generation.")
    parser.add_argument("--graph_random_size_x", type=float,
                        default=500.0,
                        help="The x-axis size of the world for random graph generation.")
    parser.add_argument("--graph_random_size_y", type=float,
                        default=500.0,
                        help="The y-axis size of the world for random graph generation.")
    parser.add_argument("--reward_method_terminal", type=str,
                        default="average", 
                        help="the method to use for terminal reward.")
    parser.add_argument("--reward_interval", type=int, default=-1,
                        help="number of steps between the periodic reward. -1 disables periodic reward")
    parser.add_argument("--agent_speed", type=float, default=10.0,
                        help="the speed of each agent")
    parser.add_argument("--action_method", type=str, default="full",
                        choices=["full", "neighbors", "neighbors_with_comm_boolean"],
                        help="the action method to use. 'neighbors_with_comm_boolean' extends "
                             "'neighbors' with a binary communication-request action: agents "
                             "may request a broadcast from all other agents, who respond by "
                             "sharing their local observations.")
    parser.add_argument("--observe_method", type=str, default="adjacency", 
                        choices=["adjacency", "coordinates", "pyg"],
                        help="the observation method to use")
    parser.add_argument("--observe_method_global", type=str, default="", 
                        help="the observation method to use for global observation")
    parser.add_argument("--observe_bitmap_size", type=int, default=50, 
                        help="the size (squared) to which the bitmap should be scaled for observation")
    parser.add_argument("--observation_radius", type=float, default=999999, 
                        help="the observable radius for each agent")
    parser.add_argument("--observation_radius_random_min", type=float, default=0.0,
                        help="the minimum random observable radius for each agent")
    parser.add_argument("--observation_radius_random_max", type=float, default=0.0,
                        help="the maximum random observable radius for each agent")
    parser.add_argument("--attrition_method", type=str, default="none", 
                        help="the method to use for agent attrition")
    parser.add_argument("--attrition_fixed_times", type=list, default=[], 
                        help="the fixed attrition times")
    parser.add_argument("--attrition_random_probability", type=float, default=0.0,
                        help="the random attrition probability")
    parser.add_argument("--attrition_min_agents", type=int, default=2,
                        help="the minimum number of agents that must be present for attrition to occur")
    parser.add_argument("--communication_model", type=str, default="none", 
                        help="the model name to use for communication. The \"none\" model indicates no comms allowed")
    parser.add_argument("--communication_probability", type=float, default=0.1, 
                        help="the probability of successful communication")
    parser.add_argument("--regenerate_graph_on_reset", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to regenerate the graph on reset.")
    parser.add_argument("--regenerate_graph_every", type=int, default=20,
                        help="The number of steps after which to regenerate the graph.")
    parser.add_argument("--max_nodes", type=int, default=50,
                        help="The maximum number of nodes in a single graph observation.")
    parser.add_argument("--max_neighbors", type=int, default=10,
                        help="The maximum number of neighbors per node in graph observations.")
    parser.add_argument("--require_explicit_visit", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to require explicit visitation of nodes.")
    parser.add_argument("--action_full_max_nodes", type=int, default=40,
                        help="The maximum number of nodes in the full action space.")
    parser.add_argument("--action_neighbors_max_degree", type=int, default=10,
                        help="The maximum degree of neighbors in the neighbors action space.")


def validate_args(parsed_args):
    ''' Validates the arguments. '''
    
    if parsed_args.graph_random:
        parsed_args.graph_name = f"random{parsed_args.graph_random_nodes}"


def env(*args, **kwargs):
    ''' Returns the environment class. '''
    return parallel_env(*args, **kwargs)


def raw_env(*args, **kwargs):
    ''' Returns the raw environment class. '''
    env = parallel_env(*args, **kwargs)
    env = parallel_to_aec(env)
    return env


class PatrolAgent():
    ''' This class stores all agent state. '''

    def __init__(self, id, position=(0.0, 0.0), speed = 1.0, observationRadius=np.inf, startingNode=None, currentState = 1, max_nodes = 50):
        self.id = id
        self.name = f"agent_{id}"
        self.startingPosition = position
        self.startingSpeed = speed
        self.startingNode = startingNode
        self.observationRadius = observationRadius
        self.currentState = currentState
        self.max_nodes = max_nodes
        self.reset()
    
    
    def reset(self):
        self.position = self.startingPosition
        self.speed = self.startingSpeed
        self.edge = None
        self.currentAction = -1.0
        self.lastNode = self.startingNode
        self.lastNodeVisited = None
     

class parallel_env(ParallelEnv):
    metadata = {
        "name": "patrolling_zoo_v0",
        "render_modes": ["human", "rgb_array"],
    }

    class OBSERVATION_CHANNELS(IntEnum):
        AGENT_ID = 0
        IDLENESS = 1
        GRAPH = 2

    def __init__(self,
                 num_agents = 3,
                 communication_model = "none",
                 communication_probability = 0.0,
                 require_explicit_visit = True,
                 agent_speed = 1.0,
                 alpha = 10.0,
                 beta = 100.0,
                 reward_comms_penalty_weight = 1.0,
                 action_method = "full",
                 action_full_max_nodes = 40,
                 action_neighbors_max_degree = 15,
                 reward_method_terminal = "average",
                 observation_radius = np.inf,
                 observation_radius_random_min = 0.0,
                 observation_radius_random_max = 0.0,
                 observe_method = "adjacency",
                 observe_method_global = "",
                 observe_bitmap_size = 50,
                 attrition_method = "none",
                 attrition_random_probability = 0.0,
                 attrition_min_agents = 2,
                 attrition_fixed_times = [],
                 max_cycles: int = -1,
                 max_nodes: int = 50,
                 max_neighbors: int = 15,
                 reward_interval: int = -1,
                 regenerate_graph_on_reset: bool = False,
                 regenerate_graph_every: int = 20,
                 graph_random = False,
                 graph_random_nodes = 40,
                 graph_random_radius = 75.0,
                 graph_random_size_x = 500.0,
                 graph_random_size_y = 500.0,
                 graph_file = os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "cumberland.graph"),
                ):
        """
        Initialize the patrolling environment.
        """
        super().__init__()

        # Configuration.
        self.requireExplicitVisit = require_explicit_visit
        self.observation_radius = observation_radius
        self.observation_radius_random_min = observation_radius_random_min
        self.observation_radius_random_max = observation_radius_random_max
        self.max_cycles = max_cycles
        self.comms_model = CommunicationModel(model=communication_model, p=communication_probability)
        self.action_method = action_method
        self.action_full_max_nodes = action_full_max_nodes
        self.action_neighbors_max_degree = action_neighbors_max_degree
        self.reward_method_terminal = reward_method_terminal
        self.observe_method = observe_method
        self.observe_method_global = observe_method_global if observe_method_global != "" else observe_method
        self.observe_bitmap_dims = (observe_bitmap_size, observe_bitmap_size)
        self.attrition_method = attrition_method
        self.attrition_random_probability = attrition_random_probability
        self.attrition_times = attrition_fixed_times
        self.attrition_min_agents = attrition_min_agents
        self.regenerate_graph_on_reset = regenerate_graph_on_reset
        self.regenerate_graph_every = regenerate_graph_every
        self.max_nodes = max_nodes
        self.max_neighbors = max_neighbors

        self.reward_interval = reward_interval

        self.alpha = alpha
        self.beta = beta
        self.reward_comms_penalty_weight = reward_comms_penalty_weight

        # Create patrol graph.
        if graph_random:
            self.pg = PatrolGraph(numNodes=graph_random_nodes, radius=graph_random_radius, sizeX=graph_random_size_x, sizeY=graph_random_size_y)
        else:
            self.pg = PatrolGraph(graph_file)

        # Create the agents with random starting positions.
        self.agentOrigins = random.sample(list(self.pg.graph.nodes), num_agents)
        startingPositions = [self.pg.getNodePosition(i) for i in self.agentOrigins]
        self.possible_agents = [
            PatrolAgent(i, startingPositions[i],
                        speed = agent_speed,
                        startingNode = self.agentOrigins[i],
                        observationRadius = self.observation_radius,
                        max_nodes = self.max_nodes
            ) for i in range(num_agents)
        ]

        # Create the action space.
        action_space = self._buildActionSpace(self.action_method)
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore

        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.

        state_space = self._buildStateSpace(self.observe_method_global)
        self.state_space = state_space
        # self.state_space = spaces.Dict({agent: state_space for agent in self.possible_agents})
        
        # Create the observation space.
        obs_space = self._buildStateSpace(self.observe_method)
        self.observation_spaces = spaces.Dict({agent: obs_space for agent in self.possible_agents}) # type: ignore

        self.reset_count = 0
        self.reset()


    def _buildActionSpace(self, action_method):
        ''' Creates a gym.spaces.* object representing the action space. '''

        if action_method == "full":
            if self.action_full_max_nodes < len(self.pg.graph):
                raise ValueError("The action space is smaller than the graph size.")
            maxNodes = self.action_full_max_nodes if self.action_full_max_nodes > 0 else len(self.pg.graph)
            return spaces.Box(
                low = 0,
                high = maxNodes - 1,
                dtype = np.int32
            )
        
        elif action_method == "neighbors":
            maxDegree = self.action_neighbors_max_degree # just use a fixed size and mask it
            return spaces.Box(
                low = 0,
                high = maxDegree - 1,
                dtype = np.int32
            )
        
        elif action_method == "neighbors_with_comm_boolean":
            maxDegree = self.action_neighbors_max_degree
            return spaces.Dict({
                "movement": spaces.Box(
                    low = 0,
                    high = maxDegree - 1,
                    dtype = np.int32
                ),
                "communication": spaces.Box(
                    low = 0,
                    high = 1,
                    dtype = np.int32
                )
            })


    def _buildStateSpace(self, observe_method):
        ''' Creates a state space given the observation method.
            Returns a gym.spaces.* object. '''
        
        # Create the state space dictionary.
        state_space = {}

        # Add to the dictionary depending on the observation method.

        # Add agent id.
        # if observe_method in ["adjacency", "coordinates"]:
        #     state_space["agent_id"] = spaces.Box(
        #         low = -1,
        #         high = len(self.possible_agents),
        #         dtype=np.int32
        #     )

        # Add vertex idleness time.
        if observe_method in ["adjacency", "coordinates"]:
            state_space["vertex_state"] = spaces.Dict({
                v: spaces.Box(
                    low = -1.0,
                    high = np.inf,
                ) for v in range(self.pg.graph.number_of_nodes())
            }) # type: ignore
        
        # Add adjacency matrix.
        if observe_method in ["adjacency"]:
            state_space["adjacency"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes()),
                dtype=np.float32,
            )
        
        # Add agent graph position vector.
        if observe_method in ["adjacency"]:
            state_space["agent_graph_position"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                    high = np.array([self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes(), 1.0], dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore

        # Add vertex 2D coordinates.
        if observe_method in ["coordinates"]:
            state_space["vertex_position"] = spaces.Dict({
                v: spaces.Box(
                    low = -np.inf,
                    high = np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ) for v in range(self.pg.graph.number_of_nodes())
            }) # type: ignore

        # Add agent 2D coordinates.
        if observe_method in ["coordinates"]:
            state_space["agent_position"] = spaces.Dict({
                a: spaces.Box(
                    low = -np.inf,
                    high = np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ) for a in self.possible_agents
            }) # type: ignore
        
        if observe_method in ["pyg"]:
            if self.action_method in ["neighbors", "neighbors_with_comm_boolean"]:
                edge_space = spaces.Box(
                    # weight, neighborID
                    low = np.array([0.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf], dtype=np.float32),
                )
                node_space = spaces.Box(
                    # nodeType, degree
                    low = np.array([-np.inf, 0.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf], dtype=np.float32),
                )
                node_type_idx = 0
            else:
                edge_space = spaces.Box(
                    # weight
                    low = np.array([0.0], dtype=np.float32),
                    high = np.array([np.inf], dtype=np.float32),
                )
                node_space = spaces.Box(
                    # ID, nodeType, lastNode, currentAction
                    low = np.array([0.0, -np.inf, -1.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                )
                node_type_idx = 1

            state_space["graph"] = spaces.Graph(
                node_space = node_space,
                edge_space = edge_space
            )
            state_space["graph"].node_type_idx = node_type_idx
        
        # Add observation radius (all observation modes).
        state_space["observation_radius"] = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

        if type(state_space) == dict:
            state_space = spaces.Dict(state_space)
        
        return state_space


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        self.reset_count += 1

        if seed != None:
            random.seed(seed)

        # Reset the graph.
        regenerateGraph = self.regenerate_graph_on_reset and self.reset_count % self.regenerate_graph_every == 0
        randomizeIds = regenerateGraph
        self.pg.reset(seed, randomizeIds=randomizeIds, regenerateGraph=regenerateGraph)

        # Randomize the observation radius.
        if self.observation_radius_random_max > self.observation_radius_random_min:
            self.observation_radius = np.random.uniform(self.observation_radius_random_min, self.observation_radius_random_max)

        # Reset the information about idleness over time.
        self.avgIdlenessTimes = []

        # Reset the node visit counts.
        self.nodeVisits = np.zeros(self.pg.graph.number_of_nodes())

        # Reset the agents.
        self.agentOrigins = random.sample(list(self.pg.graph.nodes), len(self.possible_agents))
        startingPositions = [self.pg.getNodePosition(i) for i in self.agentOrigins]
        self.agents = copy(self.possible_agents)
        for agent in self.possible_agents:
            agent.startingPosition = startingPositions[agent.id]
            agent.startingNode = self.agentOrigins[agent.id]
            agent.observationRadius = self.observation_radius
            agent.reset()
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)

        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.agents}
        
        # Set up the extra information dictionary.
        info = {
            agent: {
                "ready": True
            } for agent in self.agents
        }

        # Get the observation and visibility for each agent.
        observation = {}
        for agent in self.agents:
            obs, obs_mask = self.observe(agent)
            observation[agent] = obs
            info[agent]["visibility_mask"] = obs_mask

        return observation, info

    def get_reset_info(self):
        ''' Returns the full info dict (including global observations) for the current reset state.
            Called by the PettingZoo wrapper after reset() to populate observation_global etc.
        '''
        info = {
            agent: {
                "ready": True
            } for agent in self.agents
        }

        for agent in self.agents:
            _, obs_mask = self.observe(agent)
            info[agent]["visibility_mask"] = obs_mask
            og, vmg = self._populateStateSpace(self.observe_method_global, agent, radius=None, allow_done_agents=False)
            info[agent]["observation_global"] = og
            info[agent]["visibility_mask_global"] = vmg

        _, state_visibility_mask = self._state()
        info["state_visibility_mask"] = state_visibility_mask

        return info


    def render(self, predicted_positions = None, figsize=(12, 9), history_length=10, **kwargs):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).
                
            Returns:
                None
        '''
        self._plotted_prediction_labels = set()
        fig, ax = plt.subplots(figsize=figsize)
        markers = ['p']
        markers_done = ['X']
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']

        # Draw the graph.
        pos = nx.get_node_attributes(self.pg.graph, 'pos')
        idleness = [self.pg.getNodeIdlenessTime(i, self.step_count) for i in self.pg.graph.nodes]
        nodeColors = [self._minMaxNormalize(idleness[i], a=0.0, b=100, minimum=0.0, maximum=self.step_count) for i in self.pg.graph.nodes]
        nx.draw_networkx(self.pg.graph,
                         pos,
                         with_labels=True,
                         node_color=idleness,
                         edgecolors='black',
                         vmin=0,
                         vmax=100,
                         cmap='Purples',
                         node_size=600,
                         font_size=10,
                         font_color='black'
        )
        weights = {key: np.round(value, 1) for key, value in nx.get_edge_attributes(self.pg.graph, 'weight').items()}
        nx.draw_networkx_edge_labels(self.pg.graph, pos, edge_labels=weights, font_size=7)
        
        # Draw the agents.
        for i, agent in enumerate(self.possible_agents):
            marker = markers[i % len(markers)] if agent in self.agents else markers_done[i % len(markers_done)]
            color = colors[i % len(colors)]
            plt.scatter(*agent.position, color=color, marker=marker, zorder=10, alpha=0.3, s=300)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=agent.name, alpha=0.5)

        # Draw the predicted agent positions if provided.
        # Only supported for adjacency-type state spaces that have named agent positions.
        _prediction_supported = (
            predicted_positions is not None
            and isinstance(self.state_space, spaces.Dict)
            and "agent_graph_position" in self.state_space.spaces
        )
        if _prediction_supported:
            pred_agent0 = predicted_positions[0]
            if hasattr(pred_agent0, 'numpy'):
                pred_agent0 = pred_agent0.numpy()
            pred_unflattened = spaces.unflatten(self.state_space, pred_agent0.flatten())

            # Plot history of predictions from the perspective of agent 0.
            # state_space obs is a single-agent view (not keyed by agent at the top level).
            graph_pos = pred_unflattened["agent_graph_position"]

            for i, agent in enumerate(self.possible_agents):
                if agent in graph_pos:
                    node_id_a = int(round(graph_pos[agent][0]))
                    node_id_b = int(round(graph_pos[agent][1]))
                    if node_id_a < 0 or node_id_b < 0 or node_id_a >= self.pg.graph.number_of_nodes() or node_id_b >= self.pg.graph.number_of_nodes():
                        continue

                    posA = np.array(self.pg.graph.nodes[node_id_a]["pos"])
                    posB = np.array(self.pg.graph.nodes[node_id_b]["pos"])
                    dist = self._dist(posA, posB)
                    if dist < 1e-6:
                        prediction = posA
                    else:
                        prediction = posA + graph_pos[agent][2] * (posB - posA)

                    color = colors[i % len(colors)]
                    label = f"{agent.name} prediction"
                    if label in self._plotted_prediction_labels:
                        label = None
                    else:
                        self._plotted_prediction_labels.add(label)
                    plt.scatter(*prediction, color=color, marker="*", s=250, alpha=0.6, zorder=11)
                    plt.plot([], [], color=color, marker="*", linestyle='None', label=label)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.gcf().text(0,0,f'Current step: {self.step_count}, Average idleness time: {self.pg.getAverageIdlenessTime(self.step_count):.2f}')
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
            elif action_space.__class__.__name__ == "Box" and action_space.dtype == np.int32:
                shape = action_space.high - action_space.low + 1
                return spaces.Box(
                    low = np.zeros(shape, dtype=np.int32),
                    high = np.ones(shape, dtype=np.int32),
                    dtype = np.int32
                )
            else:
                raise NotImplementedError(f"Action space {action_space} not supported for action masking.")
        return get_available_action_space(action_space)


    def state(self):
        ''' Returns the global state of the environment.
            This is useful for centralized training, decentralized execution. '''
        
        return self._state()[0]

    def _state(self):

        return self._populateStateSpace(self.observe_method_global, self.possible_agents[0], radius=np.inf, allow_done_agents=True, global_state=True)


    def state_ALL(self):
        ''' Similar to the state_old() method, but this returns a customized copy of the state space for each agent.
            This is useful for centralized training, decentralized execution. '''
        
        state = {}
        for agent in self.possible_agents:
            state[agent] = self._populateStateSpace(self.observe_method_global, agent, radius=np.inf, allow_done_agents=True)[0]
        return state


    def observe(self, agent, radius=None, allow_done_agents=False, senders=set(), force_idleness_visible=True):
        ''' Returns the observation for the given agent.'''

        return self._populateStateSpace(self.observe_method, agent, radius, allow_done_agents, senders=senders, force_idleness_visible=force_idleness_visible)


    def available_actions(self, agent):
        ''' Returns the dictionary of available actions for all agents.
            This is not standard in the Pettingzoo API but is useful. '''
        return self.available_actions_dict[agent]


    def _populateStateSpace(self, observe_method, agent, radius, allow_done_agents, global_state=False, senders=set(), force_idleness_visible=True):
        ''' Returns a populated state/observation space.'''

        if radius == None:
            radius = agent.observationRadius

        # Determine list of allowed agents.
        if allow_done_agents:
            agentList = set(self.possible_agents)
        else:
            agentList = set(self.agents)

        # Calculate the list of visible agents and vertices.
        vertices = set()
        agents = set([agent]) | (senders & agentList)

        # Expand vertices from the ego agent's own observation radius only.
        # Senders do not communicate node idleness; only agent positions are shared.
        obs_v = set(v for v in self.pg.graph.nodes if self._dist(self.pg.getNodePosition(v), agent.position) <= radius)
        vertices |= obs_v

        # Expand visible agents from all observers (ego agent + senders).
        observers = copy(agents)
        for a in observers:
            obs_a = set(ag for ag in agentList if self._dist(ag.position, a.position) <= radius)
            agents |= obs_a
        
        agents = sorted(agents, key=lambda a: a.id)
        vertices = sorted(vertices)
        
        obs = {}
        obs_mask = {}

        # Add agent ID.
        # if observe_method in ["adjacency", "coordinates"]:
        #     obs["agent_id"] = agent.id
        #     obs_mask["agent_id"] = np.array([True], dtype=bool)

        # Add vertex idleness time (raw).
        if observe_method in ["adjacency", "coordinates"]:
            obs["vertex_state"] = {}
            obs_mask["vertex_state"] = {}

            idleness_visible = force_idleness_visible

            # Fill in actual values.
            # obs_mask is always False for vertex_state (idleness is never revealed to agents).
            for node in range(self.pg.graph.number_of_nodes()):
                obs["vertex_state"][node] = self.pg.getNodeIdlenessTime(node, self.step_count)
                obs_mask["vertex_state"][node] = np.array([idleness_visible], dtype=bool)

        # Add vertex 2D coordinates.
        if observe_method in ["coordinates"]:
            obs["vertex_position"] = {}
            obs_mask["vertex_position"] = {}

            for node in range(self.pg.graph.number_of_nodes()):
                obs["vertex_position"][node] = np.array(self.pg.getNodePosition(node), dtype=np.float32)
                obs_mask["vertex_position"][node] = np.array([True], dtype=bool)

        # Add agent 2D coordinates.
        if observe_method in ["coordinates"]:
            obs["agent_position"] = {}
            obs_mask["agent_position"] = {}

            for a in self.possible_agents:
                obs["agent_position"][a] = np.array(a.position, dtype=np.float32)
                obs_mask["agent_position"][a] = np.ones(2, dtype=bool) if a in agents else np.zeros(2, dtype=bool)

        # Add weighted adjacency matrix (normalized).
        if observe_method in ["adjacency"]:
            # Create adjacency matrix.
            adjacency = -1.0 * np.ones((self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes()), dtype=np.float32)
            maxWeight = max([self.pg.graph.edges[e]["weight"] for e in self.pg.graph.edges])
            minWeight = min([self.pg.graph.edges[e]["weight"] for e in self.pg.graph.edges])
            for edge in self.pg.graph.edges:
                weight = self._minMaxNormalize(self.pg.graph.edges[edge]["weight"], minimum=minWeight, maximum=maxWeight)
                adjacency[edge[0], edge[1]] = weight
                adjacency[edge[1], edge[0]] = weight
            obs["adjacency"] = adjacency
            obs_mask["adjacency"] = np.ones((self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes()), dtype=bool)
        
        # Add agent graph position vector.
        if observe_method in ["adjacency"]:
            graphPos = {}
            obs_mask["agent_graph_position"] = {}

            # Fill in actual values. Set the obs_mask to True for agents that are visible.
            for a in self.possible_agents:
                vec = np.zeros(3, dtype=np.float32)
                if a.edge == None:
                    vec[0] = a.lastNode
                    vec[1] = a.lastNode
                    vec[2] = 1.0
                else:
                    vec[0] = a.edge[0]
                    vec[1] = a.edge[1]
                    vec[2] = self._getAgentPathLength(a, self._getPathToNode(a, a.edge[0])) / self.pg.graph.edges[a.edge]["weight"]
                graphPos[a] = vec
                obs_mask["agent_graph_position"][a] = np.ones(3, dtype=bool) if a in agents else np.zeros(3, dtype=bool)
            obs["agent_graph_position"] = graphPos

        if observe_method in ["pyg"]:
            obs_mask = None

            # Convert vertices to a set.
            vertices = set(vertices)
            agents = set(agents)
            agents.add(agent)

            # Copy pg map to g
            g = self.pg.graph.copy()
 
            # Get a list of last visit times for each visible node.
            lastVisits = {i: self.pg.getNodeVisitTime(i) for i in vertices}
            
            # Get min and max idleness times for normalization.
            maxIdleness = self.step_count - min(lastVisits.values()) if len(lastVisits) > 0 else 0
            minIdleness = self.step_count - max(lastVisits.values()) if len(lastVisits) > 0 else 0
            allSame = maxIdleness == minIdleness

            # Set attributes of patrol graph nodes.
            # idleness_map = {}
            node_type_map = {}
            for node in g.nodes:
                # # Idleness is only observable in the global state.
                # if global_state and node in vertices:
                #     idleness_map[node] = 1.0 if allSame else self._minMaxNormalize(
                #         self.step_count - lastVisits[node], minimum=minIdleness, maximum=maxIdleness
                #     )
                # else:
                #     idleness_map[node] = -1.0
                node_type_map[node] = NODE_TYPE.OBSERVABLE_NODE if node in vertices else NODE_TYPE.UNOBSERVABLE_NODE

            nx.set_node_attributes(g, -1.0, "lastNode")
            nx.set_node_attributes(g, -1.0, "currentAction")
            # nx.set_node_attributes(g, idleness_map, "idlenessTime")
            nx.set_node_attributes(g, node_type_map, "nodeType")

            # Traverse through all visible agents and add their positions as new nodes to g
            agentnodes = set()
            for a in agents:
                # To avoid node ID conflicts, generate a unique node ID
                agent_node_id = f"agent_{a.id}_pos"
                g.add_node(
                    agent_node_id,
                    pos = a.position,
                    id = -1 - a.id,
                    nodeType = NODE_TYPE.AGENT,
                    visitTime = 0.0,
                    lastNode = g.nodes[a.lastNode]["id"] if a.lastNode in g.nodes else -1.0,
                    currentAction = a.currentAction if a in agents else -1.0
                )
                agentnodes.add(agent_node_id)

                # Check if the agent has an edge that it is currently on
                if a.edge is None:
                    # If the agent is not on an edge, add an edge from the agent's node to the node it is currently on
                    g.add_edge(agent_node_id, a.lastNode, weight=0.0)

                    # Add all of a.lastNode's neighbors as edges to the agent's node.
                    for neighbor in g.neighbors(a.lastNode):
                        if g.nodes[neighbor]["nodeType"] != NODE_TYPE.AGENT:
                            g.add_edge(agent_node_id, neighbor, weight=g.edges[(a.lastNode, neighbor)]["weight"])
                else:
                    node1_id, node2_id = a.edge

                    # Calculate weights or set them on a case-by-case basis
                    weight_to_node1 = self._calculateEdgeWeight(a.position, g.nodes[node1_id]['pos'])
                    weight_to_node2 = self._calculateEdgeWeight(a.position, g.nodes[node2_id]['pos'])

                    g.add_edge(agent_node_id, node1_id, weight=weight_to_node1)
                    g.add_edge(agent_node_id, node2_id, weight=weight_to_node2)
            
            # Normalize the edge weights of g.
            weights = nx.get_edge_attributes(g, 'weight')
            maxWeight = max(weights.values())
            minWeight = min(weights.values())
            for edge in g.edges:
                g.edges[edge]["weight"] = self._minMaxNormalize(weights[edge], minimum=minWeight, maximum=maxWeight)
            
            # Turn g into a digraph, dg
            # dg = nx.DiGraph(g)
            dg = g

            if self.action_method in ["neighbors", "neighbors_with_comm_boolean"]:
                for i in dg.nodes:
                    # Add degree to the node features.
                    dg.nodes[i]["degree"] = dg.out_degree(i)

                    # Add neighbor indices to the edges.
                    idx = 0
                    for j in dg.neighbors(i):
                        if dg.nodes[j]["nodeType"] == NODE_TYPE.AGENT:
                            dg.edges[(i, j)]["neighborIndex"] = -1
                        else:
                            dg.edges[(i, j)]["neighborIndex"] = idx
                            idx += 1

            # Trim the graph to only include the nodes and edges that are visible to the agent.
            # subgraph = nx.subgraph(dg, vertices | agentnodes)
            subgraph = dg
            subgraphNodes = list(subgraph.nodes)

            if self.action_method in ["neighbors", "neighbors_with_comm_boolean"]:
                edge_attrs = ["weight", "neighborIndex"]
                node_attrs = ["nodeType", "degree"]
                # node_attrs = ["nodeType", "idlenessTime", "degree"]
                # node_attrs = ["id", "nodeType", "idlenessTime", "lastNode", "currentAction"]
            else:
                edge_attrs = ["weight"]
                node_attrs = ["id", "nodeType", "lastNode", "currentAction"]

            # Convert g to PyG
            data = from_networkx(
                subgraph,
                group_node_attrs=node_attrs,
                group_edge_attrs=edge_attrs
            )
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()

            # Calculate the agent_mask based on the graph node ID assigned to this agent.
            if agent.edge == None:
                idx = subgraphNodes.index(agent.lastNode)
                neighborhood = list(subgraph.neighbors(agent.lastNode))
            else:
                idx = subgraphNodes.index(f"agent_{agent.id}_pos")
                neighborhood = list(subgraph.neighbors(f"agent_{agent.id}_pos"))
            agent_mask = np.zeros(data.num_nodes, dtype=bool)
            agent_mask[idx] = True
            data.agent_idx = idx
            data.agent_mask = agent_mask

            # Calculate neighbor information.
            neighbors = []
            for neighbor in neighborhood:
                if subgraph.nodes[neighbor]["nodeType"] != NODE_TYPE.AGENT:
                    neighbors.append(subgraphNodes.index(neighbor))
            nbrMask = np.zeros(self.max_nodes, dtype=bool)
            nbrMask[neighbors] = True
            data.neighbors = neighbors
            data.neighbors_mask = nbrMask

            obs["graph"] = data

        # Add observation radius (all observation modes).
        obs["observation_radius"] = np.array([agent.observationRadius], dtype=np.float32)
        if obs_mask is not None:
            obs_mask["observation_radius"] = np.array([True], dtype=bool)

        if (type(obs) == dict and obs == {}) or (type(obs) != dict and len(obs) < 1):
            raise ValueError(f"Invalid observation method {observe_method}")

        # If getting the global state, set the fixed/visible mask to 0 for everything
        if global_state:
            if observe_method in ["adjacency", "coordinates"]:
                for key in obs_mask:
                    if type(obs_mask[key]) == dict:
                        for subkey in obs_mask[key]:
                            obs_mask[key][subkey] = np.zeros_like(obs_mask[key][subkey], dtype=bool)
                    else:
                        obs_mask[key] = np.zeros_like(obs_mask[key], dtype=bool)

        return obs, obs_mask
    
    def _calculateEdgeWeight(self, pos1, pos2):
        '''Calculate the weights of the edges based on the position of the two points, here simply use the Euclidean distance'''
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

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

        # Track communication requests made this step: {requesting_agent: set_of_sender_agents}.
        comms_requests = {}
        info_dict["communication/requests_made"] = 0

        # Perform actions.
        for agent in self.agents:
            if agent in action_dict:
                action = action_dict[agent]
                action = spaces.unflatten(self.action_space(agent), action)

                # Check if the action is valid.
                if not self.action_space(agent).contains(action):
                    raise ValueError(f"Invalid action {action} of type {type(action)} provided.")

                # Get the agent's movement action.
                if self.action_method in ["full", "neighbors"]:
                    action_movement = int(action)
                elif self.action_method == "neighbors_with_comm_boolean":
                    action_movement = int(action["movement"])

                # Store this as the agent's last movement action.
                agent.currentAction = action_movement

                # Get the destination node.
                dstNode = self.getDestinationNode(agent, action_movement)
                
                # Calculate the shortest path.
                path = self._getPathToNode(agent, dstNode)
                pathLen = self._getAgentPathLength(agent, path)
                
                # Take a step towards the next node.
                stepSize = np.random.normal(loc=agent.speed, scale=1.0)
                for nextNode in path:
                    reached, stepSize = self._moveTowardsNode(agent, nextNode, stepSize)

                    # The agent has reached the next node.
                    if reached:
                        if nextNode == dstNode or not self.requireExplicitVisit:
                            # The agent has reached its destination, visiting the node.
                            # The agent receives a reward for visiting the node.
                            r = self.onNodeVisit(nextNode, self.step_count)
                            reward_dict[agent] += r

                            agent.lastNodeVisited = nextNode
                            if nextNode == dstNode:
                                agent.currentAction = -1.0
                                info_dict[agent]["ready"] = True
                        # Agent reached the destination, assign a new speed from normal distribution
                        # agent.speed = max(np.random.normal(loc=agent.startingSpeed, scale=5.0), 1.0)
            
                    # The agent has exceeded its movement budget for this step.
                    if stepSize <= 0.0:
                        break
                
                # Handle communication.
                if self.action_method in ["full", "neighbors"]:
                    # Hack - we are doing broadcast-based requests, so always receive from all agents.
                    if self.comms_model.canReceive(None, agent):
                        senders = set(other for other in self.agents if other is not agent)
                        if senders:
                            comms_requests[agent] = senders
                            info_dict["communication/requests_made"] += 1
                elif self.action_method == "neighbors_with_comm_boolean":
                    action_movement = int(action["movement"])
                    action_communication = bool(action["communication"])
                    if action_communication:
                        senders = set(other for other in self.agents if other is not agent)
                        if senders:
                            comms_requests[agent] = senders
                            info_dict["communication/requests_made"] += 1
                        reward_dict[agent] += -1.0 * self.reward_comms_penalty_weight

        # Record the average idleness time at this step.
        avg = self._minMaxNormalize(self.pg.getAverageIdlenessTime(self.step_count), minimum=0.0, maximum=self.step_count)
        self.avgIdlenessTimes.append(avg)        

        # Perform observations.
        for agent in self.possible_agents:
            # Determine which agents have sent an observation via comms.
            senders = comms_requests.get(agent, set())

            # Local observation.
            obs, obs_mask = self.observe(agent, senders=senders)
            obs_dict[agent] = obs
            info_dict[agent]["visibility_mask"] = obs_mask

            # Global observation.
            og, vmg = self._populateStateSpace(self.observe_method_global, agent, radius=None, allow_done_agents=False, senders=senders)
            info_dict[agent]["observation_global"] = og
            info_dict[agent]["visibility_mask_global"] = vmg

        # Add the state fixed mask. (Nothing is fixed.)
        _, state_visibility_mask = self._state()
        info_dict["state_visibility_mask"] = state_visibility_mask
        
        # Record miscellaneous information.
        for i, n in enumerate(self.nodeVisits):
            info_dict[f"node_visits/node_{i}"] = n
        info_dict["avg_idleness"] = self.pg.getAverageIdlenessTime(self.step_count)
        info_dict["stddev_idleness"] = self.pg.getStdDevIdlenessTime(self.step_count)
        info_dict["worst_idleness"] = self.pg.getWorstIdlenessTime(self.step_count)
        info_dict["agent_count"] = len(self.agents)

        # Check truncation conditions.
        if lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles):
            for agent in self.agents:
                # Provide an end-of-episode reward.
                if self.reward_method_terminal == "average":
                    reward_dict[agent] += self.beta * self.step_count / (self.pg.getAverageIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "worst":
                    reward_dict[agent] += self.beta * self.step_count / (self.pg.getWorstIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "stddev":
                    reward_dict[agent] += self.beta * self.step_count / (self.pg.getStdDevIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "averageAverage":
                    avg = np.average(self.avgIdlenessTimes)
                    # reward_dict[agent] += self.beta * self.step_count / (avg + 1e-8)
                    reward_dict[agent] -= self.beta * avg
                elif self.reward_method_terminal == "divNormalizedWorst":
                    reward_dict[agent] /= self._minMaxNormalize(self.pg.getWorstIdlenessTime(self.step_count), minimum=0.0, maximum=self.max_cycles)
                elif self.reward_method_terminal != "none":
                    raise ValueError(f"Invalid terminal reward method {self.reward_method_terminal}")

                info_dict[agent]["ready"] = True
            
                truncated_dict[agent] = True
            self.agents = []
        
        # Provide a reward at a fixed interval.
        elif self.reward_interval >= 0 and self.step_count % self.reward_interval == 0:
            for agent in self.agents:
                # reward_dict[agent] += self.beta * self.step_count / (self.pg.getAverageIdlenessTime(self.step_count) + 1e-8)
                reward_dict[agent] -= self.beta * self._minMaxNormalize(self.pg.getAverageIdlenessTime(self.step_count), minimum=0.0, maximum=self.step_count)

        done_dict = {agent: self.dones[agent] for agent in self.possible_agents}

        # Set available actions.
        self.available_actions_dict = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def onNodeVisit(self, node, timeStamp):
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''
        
        # Record the node visit.
        self.nodeVisits[node] += 1

        # Calculate a visitation reward.
        idleness = self.pg.getNodeIdlenessTime(node, timeStamp)
        reward = self._minMaxNormalize(idleness, minimum=0.0, maximum=max(timeStamp, 1))
        reward = self.alpha * reward

        # Update the node visit time.
        self.pg.setNodeVisitTime(node, timeStamp)

        return reward


    def getDestinationNode(self, agent, action):
        ''' Returns the destination node for the given agent and action. '''

        # Interpret the action using the "full" method.
        if self.action_method == "full":
            if action not in self.pg.graph.nodes:
                raise ValueError(f"Invalid action {action} for agent {agent.name}")
            dstNode = action
        
        # Interpret the action using the "neighbors" method.
        elif self.action_method in ["neighbors", "neighbors_with_comm_boolean"]:
            if agent.edge == None:
                if action >= self.pg.graph.out_degree(agent.lastNode):
                    raise ValueError(f"Invalid action {action} for agent {agent.name}. Node {agent.lastNode} has only {self.pg.graph.out_degree(agent.lastNode)} neighbors.")
                dstNode = list(self.pg.graph.successors(agent.lastNode))[action]
            else:
                if action != agent.currentAction:
                    raise ValueError(f"Invalid action {action} for agent {agent.name}. Must complete action {agent.currentAction} first.")
                dstNode = list(self.pg.graph.neighbors(agent.lastNode))[action]
        
        else:
            raise ValueError(f"Invalid action method {self.action_method}")
        
        return dstNode


    def _moveTowardsNode(self, agent, node, stepSize):
        ''' Takes a single step towards the next node.
            Returns a tuple containing whether the agent has reached the node
            and the remaining step size. '''

        # Take a step towards the next node.
        posNextNode = self.pg.getNodePosition(node)
        distCurrToNext = self._dist(agent.position, posNextNode)
        reached = distCurrToNext <= stepSize
        step = distCurrToNext if reached else stepSize
        if distCurrToNext > 0.0:
            agent.position = (
                agent.position[0] + (posNextNode[0] - agent.position[0]) * step / distCurrToNext,
                agent.position[1] + (posNextNode[1] - agent.position[1]) * step / distCurrToNext
            )
        
        # Set information about the node/edge which the agent is currently on.
        if reached:
            agent.lastNode = node
            agent.edge = None
        elif agent.lastNode != node:
            # Ensure that ordering is always the same for the edge.
            agent.edge = tuple(sorted((agent.lastNode, node)))

        return reached, max(stepSize - distCurrToNext, 0.0)


    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.sqrt(np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2))
    

    def _getPathToNode(self, agent, dstNode):
        ''' Determines the shortest path for the agent to reach the given node. '''

        path = []

        # The agent is on an edge, so determine which connected node results in shortest path.
        if agent.edge != None:
            path1 = nx.shortest_path(self.pg.graph, source=agent.edge[0], target=dstNode, weight='weight')
            pathLen1 = self._getAgentPathLength(agent, path1)
            path2 = nx.shortest_path(self.pg.graph, source=agent.edge[1], target=dstNode, weight='weight')
            pathLen2 = self._getAgentPathLength(agent, path2)
            path = path1
            if pathLen2 < pathLen1:
                path = path2
        
        # The agent is on a node. Simply calculate the shortest path.
        else:
            path = nx.shortest_path(self.pg.graph, source=agent.lastNode, target=dstNode, weight='weight')

            # Remove the first node from the path if the destination is different than the current node.
            if agent.lastNode != dstNode:
                path = path[1:]
        
        return path


    def _getAgentPathLength(self, agent, path):
        ''' Calculates the length of the given path for the given agent. '''

        pathLen = 0.0
        pathLen += self._dist(agent.position, self.pg.getNodePosition(path[0]))
        pathLen += nx.path_weight(self.pg.graph, path, weight='weight')

        return pathLen


    def _minMaxNormalize(self, x, eps=1e-8, a=0.0, b=1.0, maximum=None, minimum=None):
        ''' Normalizes numpy array x to be between a and b. '''

        if maximum is None:
            maximum = np.max(x)
        if minimum is None:
            minimum = np.min(x)
        return a + (x - minimum) * (b - a) / (maximum - minimum + eps)


    def _getAvailableActions(self, agent):
        ''' Returns the available actions for the given agent. '''

        if self.action_method == "full":
            num_nodes = self.action_space(agent).high - self.action_space(agent).low + 1
            if agent.edge == None:
                # All actions available.
                actionMap = np.zeros(num_nodes, dtype=np.float32)
                actionMap[:self.pg.graph.number_of_nodes()] = 1.0
                return actionMap
            else:
                # Only the current action available (as it is still incomplete).
                actionMap = np.zeros(num_nodes, dtype=np.float32)
                actionMap[agent.currentAction] = 1.0
                return actionMap
        
        elif self.action_method == "neighbors":
            num_nodes = self.action_space(agent).high - self.action_space(agent).low + 1
            if agent.edge == None:
                # All neighbors of the current node are available.
                actionMap = np.zeros(num_nodes, dtype=np.float32)
                # numNeighbors = self.pg.graph.out_degree(agent.lastNode) - 1 # subtract 1 since the self loop adds 2 to the degree
                numNeighbors = self.pg.graph.out_degree(agent.lastNode)
                actionMap[:numNeighbors] = 1.0
                return actionMap
            else:
                # Only the current action available (as it is still incomplete).
                actionMap = np.zeros(num_nodes, dtype=np.float32)
                actionMap[agent.currentAction] = 1.0
                return actionMap
            
        elif self.action_method == "neighbors_with_comm_boolean":
            num_moves = self.action_space(agent)["movement"].high - self.action_space(agent)["movement"].low + 1
            if agent.edge == None:
                # All neighbors of the current node are available, and communication is always available.
                actionMap = {
                    "movement": np.zeros(num_moves, dtype=np.float32),
                    "communication": np.array([1.0, 1.0], dtype=np.float32)
                }
                numNeighbors = self.pg.graph.out_degree(agent.lastNode)
                actionMap["movement"][:numNeighbors] = 1.0
            else:
                # Only the current movement action is available (as it is still incomplete), but communication is still available.
                actionMap = {
                    "movement": np.zeros(num_moves, dtype=np.float32),
                    "communication": np.array([1.0, 1.0], dtype=np.float32)
                }
                actionMap["movement"][agent.currentAction] = 1.0
            
            # Flatten the action map.
            flat = spaces.flatten(self.available_actions_space(agent), actionMap)
            return flat
        else:
            raise ValueError(f"Invalid action method {self.action_method}")