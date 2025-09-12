import numpy as np
from enum import IntEnum

class ENTITY_TYPE(IntEnum):
    AGENT = 0
    DEPOT = 1
    ZONE = 2


class AGENT_ROLE(IntEnum):
    PROSPECTOR = 0
    EXTRACTOR = 1
    HAULER = 2
    # PROSPECTOREXTRACTOR = 3


class CAP(IntEnum):
    PROSPECT = 0
    EXTRACT = 1
    CARRY = 2
    CARRY_CAPACITY = 3


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
    role = None # To be set by subclasses

    def __init__(self, world_dims, *args, speed_max = 1.0, observation_radius=np.inf, **kwargs):
        self.world_dims = world_dims
        self.speed_max = speed_max
        self.observation_radius = observation_radius
        self.capabilities = {
            CAP.PROSPECT: False,
            CAP.EXTRACT: False,
            CAP.CARRY: False,
        }
        super().__init__(*args, **kwargs)


    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.cargo = {}
        self.steps_stationary = 0
        self.mask_observed = np.zeros(self.world_dims, dtype=bool)
        self.mask_resources_observed = np.zeros(self.world_dims, dtype=bool)


# --- New agent Type subclasses ---
class Prospector(Agent):
    """Prospector: observe resources.
    """

    role = AGENT_ROLE.PROSPECTOR
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities.update({
            CAP.PROSPECT: True,
            CAP.EXTRACT: False,
            CAP.CARRY: False,
        })


class Extractor(Agent):
    """Extractor: excavates resources."""

    role = AGENT_ROLE.EXTRACTOR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities.update({
            CAP.PROSPECT: False,
            CAP.EXTRACT: True,
            CAP.CARRY: False,
        })


# class ProspectorExtractor(Agent):
#     """ProspectorExtractor: observe and extract resources.
#     """

#     role = AGENT_ROLE.PROSPECTOREXTRACTOR
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.capabilities.update({
#             CAP.PROSPECT: True,
#             CAP.EXTRACT: True,
#             CAP.CARRY: False,
#         })


class Hauler(Agent):
    """Hauler: transports resources to depots."""

    role = AGENT_ROLE.HAULER

    def __init__(self, *args, carry_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities.update({
            CAP.PROSPECT: False,
            CAP.EXTRACT: False,
            CAP.CARRY: True,
            CAP.CARRY_CAPACITY: carry_capacity
        })


class SuperBot(Agent):
    """ SuperBot: It can do anything! (TM) """

    role = AGENT_ROLE.PROSPECTOR

    def __init__(self, *args, carry_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities.update({
            CAP.PROSPECT: True,
            CAP.EXTRACT: True,
            CAP.CARRY: True,
            CAP.CARRY_CAPACITY: carry_capacity
        })


class Depot(Entity):
    ''' This base class stores all generic depot state. '''

    entity_type = ENTITY_TYPE.DEPOT

    def __init__(self, resource, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store the resource associated with this depot.
        self.resource = resource
    
    def reset(self, *args, **kwargs):
        ''' Resets the depot to its initial state. '''
        super().reset(*args, **kwargs)
        
        self.stock = 0.0
    

    @property
    def resource_id(self):
        ''' Returns the ID of the resource associated with this depot. '''
        return self.resource.resource_id


    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        # Clear delivered stock at episode start
        self.stock = 0.0


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