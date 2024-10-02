import numpy as np
from enum import IntEnum

class ENTITY_TYPE(IntEnum):
    SHEPHERD = 0
    SHEEP = 1
    ZONE_KEEPOUT = 2
    ZONE_GOAL = 3


class Entity():
    ''' This base class stores all generic entity state. '''

    def __init__(self, graph, position=None):
        self.graph = graph
        if position is None:
            position = self.get_random_start_position()
        self.startingPosition = np.copy(position)
        self._position = position
        self._velocity = np.array([0.0, 0.0])
        self.isGraphEntity = False
        if self.entity_type != None and self.entity_type in ENTITY_TYPE:
            self.isGraphEntity = True
            self.id = self.graph.getNextEntityID()
        else:
            self.id = -1
        self.reset()
    

    def __del__(self):
        if self.isGraphEntity and self.graph.hasEntity(self):
            self.graph.removeEntity(self)
    

    def reset(self, reset_start_position=False):
        ''' Resets the entity to its initial state. '''
        if reset_start_position:
            self.startingPosition = self.get_random_start_position()
        
        if self.isGraphEntity and not self.graph.hasEntity(self):
            self.graph.addEntity(self, self.startingPosition)

        self.position = np.copy(self.startingPosition)
        self.velocity = np.array([0.0, 0.0])


    def get_random_start_position(self):
        ''' Uses the graph to generate a random start position. '''

        raise NotImplementedError("get_random_start_position() must be implemented by the subclass.")


    @property
    def position(self):
        ''' Return the current position. '''
        return self._position


    @position.setter
    def position(self, pos):
        ''' Sets the agent position. '''
        self._position = pos
        if self.isGraphEntity:
            self.graph.setNodePosition(self, pos)


    @property
    def velocity(self):
        ''' Return the current velocity. '''
        return self._velocity


    @velocity.setter
    def velocity(self, vel):
        ''' Sets the agent velocity. '''
        self._velocity = vel
        if self.isGraphEntity:
            self.graph.setNodeVelocity(self, vel)


class Agent(Entity):
    ''' This base class stores all generic agent state. '''

    def __init__(self, *args, maxSpeed = 1.0, observationRadius=np.inf, currentState = 1, **kwargs):
        self.name = f"{self.__class__.__name__.lower()}_{id}"
        self.observationRadius = observationRadius
        self.currentState = currentState
        self.maxSpeed = maxSpeed
        super().__init__(*args, **kwargs)


    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.currentAction = -1.0
        
        # Map of node to list[pos: ndarray, vel: ndarray]
        self.stateBelief = {}


class Shepherd(Agent):
    ''' This class stores all shepherd state. '''

    entity_type = ENTITY_TYPE.SHEPHERD

    def get_random_start_position(self):
        return self.graph.start.get_random_point_boundary()


class Sheep(Agent):
    ''' This class stores all sheep state. '''

    entity_type = ENTITY_TYPE.SHEEP

    def get_random_start_position(self):
        return self.graph.start.get_random_point()


class CircularZone(Entity):
    ''' Position and radius information for zones '''
    def __init__(self, *args, r, entityType):
        self.entity_type = entityType
        super().__init__(*args)
        self.radius = r
        
    def contains(self, pos):
        return True if np.linalg.norm(self.position - pos) <= self.radius else False
    
    def closest_boundary(self, pos):
        ''' Returns the smallest vector from pos to the zone boundary '''
        diff = self.position - pos
        return diff * (1 - self.radius / np.linalg.norm(diff))

    def get_random_point(self):
        ''' Returns a random point within the zone. '''

        # Generate a random angle and radius.
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, self.radius)

        # Calculate the point.
        x = self.position[0] + radius * np.cos(angle)
        y = self.position[1] + radius * np.sin(angle)

        return np.array([x, y])
    
    def get_random_point_boundary(self):
        ''' Returns a random point on the boundary of the circle. '''

        # Generate a random angle.
        angle = np.random.uniform(0, 2 * np.pi)

        # Calculate the point.
        x = self.position[0] + self.radius * np.cos(angle)
        y = self.position[1] + self.radius * np.sin(angle)

        return np.array([x, y])