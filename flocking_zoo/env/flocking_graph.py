import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import random

from flocking_zoo.env.entity import CircularZone, ENTITY_TYPE

class FlockingGraph():
    ''' This class tracks agents and other entities in the flocking environment. '''
    
    def __init__(self, filepath = None, generate_random_graph=True, num_danger_zones=3, grid_size = 20, grid_border = 10, world_dimensions = np.array([100.0, 100.0])):

        self.num_danger_zones = num_danger_zones
        self.grid_size = grid_size
        self.grid_border = grid_border
        
        self.graph = nx.DiGraph()
        self.world_dimensions = world_dimensions
        self.entities = {}
        self.nextEntityID = 0
        if filepath is None:
            if generate_random_graph:
                self.generate_random_env_grid()
            else:
                self.initialize_example()
        else:
            self.loadFromFile(filepath)


    def reset(self, regenerateGraph=False):
        ''' Resets the graph to initial state.
            If regenerateGraph is True, a new random graph is generated. '''

        if regenerateGraph:
            self.entities = {}
            self.graph = nx.DiGraph()
            self.generate_random_env_grid()


    def loadFromFile(self, filepath: str):
        ''' Loads a graph from a file. '''

        raise NotImplementedError("Loading from file is not implemented yet.")

        
    def initialize_example(self):
        ''' Hard-coded example '''
        self.goal = CircularZone(self, (10, 50), r=10, entityType=ENTITY_TYPE.ZONE_GOAL)
        self.start = CircularZone(self, (70, 50), r=20, entityType=None)
        self.danger_zones = [
            CircularZone(self, (30, 50), r=10, entityType=ENTITY_TYPE.ZONE_KEEPOUT),
            CircularZone(self, (5, 60), r=5, entityType=ENTITY_TYPE.ZONE_KEEPOUT),
        ]    

    def generate_random_env_grid(self):
        ''' Uses a discrete grid for placement of zones to allow for fast non-overlapping generation. '''

        # Set up grid.
        grid_size = self.grid_size
        grid_border = (grid_size // 2) + self.grid_border
        grid_dimensions = np.ceil((self.world_dimensions - 2 * grid_border) / grid_size).astype(int)
        grid_available = np.ones(grid_dimensions, dtype=bool)
        radius_max = grid_size // 2

        # Generate random goal zone.
        goal_radius = radius_max
        grid_indices_available = np.argwhere(grid_available == True)
        grid_index = grid_indices_available[np.random.choice(np.arange(grid_indices_available.shape[0]))]
        goal_position = grid_index * grid_size + grid_border
        grid_available[tuple(grid_index)] = False
        self.goal = CircularZone(self, goal_position, r=goal_radius, entityType=ENTITY_TYPE.ZONE_GOAL)

        # Generate random non-overlapping start zone.
        start_radius = radius_max
        grid_indices_available = np.argwhere(grid_available == True)
        grid_index = grid_indices_available[np.random.choice(np.arange(grid_indices_available.shape[0]))]
        start_position = grid_index * grid_size + grid_border
        grid_available[tuple(grid_index)] = False
        self.start = CircularZone(self, start_position, r=start_radius, entityType=None)

        # Generate random danger zones.
        self.danger_zones = []
        num_danger_zones = self.num_danger_zones
        for _ in range(num_danger_zones):
            danger_radius = radius_max
            grid_indices_available = np.argwhere(grid_available == True)
            grid_index = grid_indices_available[np.random.choice(np.arange(grid_indices_available.shape[0]))]
            danger_position = grid_index * grid_size + grid_border
            grid_available[tuple(grid_index)] = False
            self.danger_zones.append(CircularZone(self, danger_position, r=danger_radius, entityType=ENTITY_TYPE.ZONE_KEEPOUT))


    def closest_danger(self, pos):
        ''' Returns the smallest vector from the position to a danger zone boundary '''
        closest = None
        closest_length = -1
        for danger_zone in self.danger_zones:
            test = danger_zone.closest_boundary(pos)
            test_length = np.linalg.norm(test)
            if closest is None or test_length < closest_length:
                closest = test
                closest_length = test_length
                
        return closest
    

    def count_in_goal(self, pos_list):
        ''' Returns count of positions inside the goal '''
        count = 0
        for pos in pos_list:
            if self.goal.contains(pos):
                count += 1
                
        return count


    def getNextEntityID(self):
        ''' Returns the next entity ID. '''

        id = self.nextEntityID
        self.nextEntityID += 1
        return id


    def hasEntity(self, entity):
        ''' Determines whether entity is a node in the graph. '''

        return entity in self.graph.nodes


    def addEntity(self, entity, pos):
        ''' Adds an entity to the graph. '''

        if entity.entity_type not in ENTITY_TYPE:
            ValueError(f"Invalid entity type {entity.entity_type}.")

        # Add node.
        self.graph.add_node(
            entity,
            pos=pos,
            vel=np.array([0.0, 0.0]),
            vel_x=0.0,
            vel_y=0.0,
            pos_x=pos[0],
            pos_y=pos[1],
            type=entity.entity_type
        )

        # Add to entity list.
        if entity.entity_type not in self.entities:
            self.entities[entity.entity_type] = []
        self.entities[entity.entity_type].append(entity)

        # Add an edge to all existing nodes.
        for node in self.graph.nodes:
            if node != entity:
                self.graph.add_edge(
                    node,
                    entity,
                    weight=self.calculateEdgeWeight(node, entity),
                    theta=self.calculateEdgeAngle(node, entity)
                )
                self.graph.add_edge(
                    entity,
                    node,
                    weight=self.calculateEdgeWeight(entity, node),
                    theta=self.calculateEdgeAngle(entity, node)
                )


    def removeEntity(self, entity):
        ''' Removes an entity from the graph. '''

        self.graph.remove_node(entity)
        self.entities[entity.entity_type].remove(entity)


    def getNodePosition(self, node):
        ''' Returns the node position as a ndarray (x, y). '''

        return self.graph.nodes[node]["pos"]
    

    def setNodePosition(self, node, pos):
        ''' Sets the node position. '''

        self.graph.nodes[node]["pos"] = np.array(pos)
        self.graph.nodes[node]["pos_x"] = pos[0]
        self.graph.nodes[node]["pos_y"] = pos[1]

        # Update edge weights.
        for neighbor in self.graph.neighbors(node):
            weight = self.calculateEdgeWeight(node, neighbor)
            self.graph[node][neighbor]["weight"] = weight
            self.graph[neighbor][node]["weight"] = weight
            self.graph[node][neighbor]["theta"] = self.calculateEdgeAngle(node, neighbor)
            self.graph[neighbor][node]["theta"] = self.calculateEdgeAngle(neighbor, node)
        
    def getNodeVelocity(self, node):
        ''' Returns the node velocity as a ndarray (x, y). '''

        return self.graph.nodes[node]["vel"]
    

    def setNodeVelocity(self, node, vel):
        ''' Sets the node velocity. '''

        self.graph.nodes[node]["vel"] = vel
        self.graph.nodes[node]["vel_x"] = vel[0]
        self.graph.nodes[node]["vel_y"] = vel[1]
    

    def getNearestNode(self, pos, epsilon=None):
        ''' Returns the nearest node to the given position.
            If epsilon is not None and no node is within epsilon, returns None. '''
        
        # Find the nearest node.
        bestDist = math.sqrt(math.pow(self.graph.nodes[0]["pos"][0] - pos[0], 2) + math.pow(self.graph.nodes[0]["pos"][1] - pos[1], 2))
        bestNode = 0
        for i in range(len(self.graph.nodes)):
            dist = math.sqrt(math.pow(self.graph.nodes[i]["pos"][0] - pos[0], 2) + math.pow(self.graph.nodes[i]["pos"][1] - pos[1], 2))
            if dist < bestDist:
                bestDist = dist
                bestNode = i

        # Check if the nearest node is within epsilon.
        if epsilon is not None and bestDist > epsilon:
            return None
        else:
            return bestNode
    

    def calculateEdgeAngle(self, node1, node2):
        ''' Calculates the angle of the edge between two nodes. '''

        diff = self.graph.nodes[node2]["pos"] - self.graph.nodes[node1]["pos"]
        angle = np.arctan2(diff[1], diff[0])
        angle = angle % (2 * np.pi)
        return angle


    def calculateEdgeWeight(self, node1, node2):
        ''' Calculates edge weight, taking into account the entity types. '''

        weight = 1.0 / (1 + 10 * self._dist(self.graph.nodes[node1]["pos"], self.graph.nodes[node2]["pos"]))
        # if hasattr(node1, "radius"):
        #     dist -= node1.radius
        # if hasattr(node2, "radius"):
        #     dist -= node2.radius
        return weight

    
    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.linalg.norm(pos1 - pos2)


    def getPyTorchGeometricGraph(self):
        ''' Returns a torch_geometric (PyG) graph object. '''

        from torch_geometric.utils.convert import from_networkx
        return from_networkx(self.graph, group_node_attrs=["pos"], group_edge_attrs=["weight"])


    def exportToFile(self, filename):
        ''' Exports to a file of the same format as `importFromFile`. '''

        raise NotImplementedError("Exporting to file is not implemented yet.")