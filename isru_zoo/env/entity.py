import numpy as np
from enum import IntEnum

class ENTITY_TYPE(IntEnum):
    AGENT = 0
    DEPOT = 1
    ZONE = 2


class Entity():
    ''' This base class stores all generic entity state. '''

    entity_count = 0

    def __init__(self, position=None):

        self.entity_id = Entity.entity_count
        self.name = f"{self.__class__.__name__.lower()}_{self.entity_id}"
        Entity.entity_count += 1

        if position is None:
            position = self.get_random_start_position()
        self.startingPosition = np.copy(position)
        self._position = position
        self._velocity = np.zeros_like(position)
        self.reset()
    

    def reset(self, reset_start_position=False, position=None):
        ''' Resets the entity to its initial state. '''
        if reset_start_position:
            if position is None:
                self.startingPosition = self.get_random_start_position()
            else:
                self.startingPosition = position
        
        self.position = np.copy(self.startingPosition)
        self.velocity = np.zeros_like(self.startingPosition)


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


    @property
    def velocity(self):
        ''' Return the current velocity. '''
        return self._velocity


    @velocity.setter
    def velocity(self, vel):
        ''' Sets the agent velocity. '''
        self._velocity = vel
    

    @property
    def id(self):
        ''' Returns the entity ID. '''
        return self.entity_id
    

    def __repr__(self):
        ''' Returns a string representation of the entity. '''
        return self.name



class Agent(Entity):
    ''' This base class stores all generic agent state. '''

    entity_type = ENTITY_TYPE.AGENT

    def __init__(self, *args, speed_max = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed_max = speed_max
        # Ensure all agents have a capabilities dict by default
        self.capabilities = {}

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)


# --- New agent Type subclasses ---
class Prospector(Agent):
    """Prospector: observe resources.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "prospector"
        # Defaults specific to prospector
        self.capabilities.update({"see_resources": True})


class Extractor(Agent):
    """Extractor: excavates resources."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "extractor"
        self.capabilities.update({"excavate": True})


class Hauler(Agent):
    """Hauler: transports resources to depots."""
    def __init__(self, *args, carry_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "hauler"
        self.capabilities.update({"carry": True, "carry_capacity": carry_capacity})


class Depot(Entity):
    ''' This base class stores all generic depot state. '''

    entity_type = ENTITY_TYPE.DEPOT

    def __init__(self, resource, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store the resource associated with this depot.
        self.resource = resource
    

    @property
    def resource_id(self):
        ''' Returns the ID of the resource associated with this depot. '''
        return self.resource.resource_id


    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)


class CircularZone(Entity):
    ''' Position and radius information for zones '''
    def __init__(self, *args, radius, entity_type, **kwargs):
        self.entity_type = entity_type
        super().__init__(*args, **kwargs)
        self.radius = radius
        
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


class GoalZone(CircularZone):
    ''' Position and radius information for goal zones '''
    entity_type = ENTITY_TYPE.ZONE

    class GOAL_STATE(IntEnum):
        UNREACHED = 0
        REACHED_ADVERSARY = 1
        REACHED_AGENT = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, entity_type=self.entity_type, **kwargs)
    
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.state = GoalZone.GOAL_STATE.UNREACHED
    
    def reached_check(self, entity):
        ''' Returns True and sets state if the entity is within the zone. '''
        if self.state != GoalZone.GOAL_STATE.UNREACHED:
            return False
        if self.contains(entity.position):
            if isinstance(entity, Adversary):
                self.state = GoalZone.GOAL_STATE.REACHED_ADVERSARY
            elif isinstance(entity, Agent):
                self.state = GoalZone.GOAL_STATE.REACHED_AGENT
            return True
        return False