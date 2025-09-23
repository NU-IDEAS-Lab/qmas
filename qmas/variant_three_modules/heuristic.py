
import numpy as np
from isru_zoo.env.entity import *
from isru_zoo.env.isru_env import parallel_env_map_obs

def get_movement_action_heuristic(observation):
    ''' Uses a simple heuristic to determine the action for the given agent based on the observation. '''

    # Just move to the bottom-right corner for now as an example.
    return parallel_env_map_obs._velocity_to_moore_index(None, np.array([1, 1]))

    def get_agent_characteristics():
        ''' Gets agent position and capabilities from the observation. '''

        # Position is where both relative pos x and y are zero.
        relative_pos_x = observation[parallel_env_map_obs.MAP_LAYERS.RELATIVE_POS_X]
        relative_pos_y = observation[parallel_env_map_obs.MAP_LAYERS.RELATIVE_POS_Y]
        agent_pos = np.argwhere((relative_pos_x == 0) & (relative_pos_y == 0))
        if agent_pos.shape[0] != 1:
            raise ValueError("Could not determine unique agent position from observation.")
        agent_pos = agent_pos[0]

        # Determine role based on agent layers.
        if observation[parallel_env_map_obs.MAP_LAYERS.AGENTS_EXTRACTOR][tuple(agent_pos)] > 0:
            capabilities = {CAP.EXTRACT: True, CAP.CARRY: False, CAP.PROSPECT: False}
        elif observation[parallel_env_map_obs.MAP_LAYERS.AGENTS_HAULER][tuple(agent_pos)] > 0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: True, CAP.PROSPECT: False}
        elif observation[parallel_env_map_obs.MAP_LAYERS.AGENTS_PROSPECTOR][tuple(agent_pos)] > 0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: False, CAP.PROSPECT: True}
        elif observation[parallel_env_map_obs.MAP_LAYERS.AGENTS_SUPERBOT][tuple(agent_pos)] > 0:
            capabilities = {CAP.EXTRACT: True, CAP.CARRY: True, CAP.PROSPECT: True}
        else:
            raise ValueError("Could not determine agent capabilities from observation.")

        return agent_pos, capabilities

    def obs_nearest_resource(pos):
        ''' Returns the direction to the nearest resource in the observation. '''
        layer = observation[self.MAP_LAYERS.RESOURCES_EXTANT]
        positions = np.argwhere(layer > 0)
        if positions.shape[0] == 0:
            return None
        dists = np.linalg.norm(positions - np.array(pos), axis=1)
        nearest_pos = positions[np.argmin(dists)]
        direction = nearest_pos - pos
        return direction

    def obs_nearest_depot(pos):
        ''' Returns the direction to the nearest depot in the observation. '''
        layer = observation[self.MAP_LAYERS.DEPOTS]
        positions = np.argwhere(layer > 0)
        if positions.shape[0] == 0:
            return None
        dists = np.linalg.norm(positions - np.array(pos), axis=1)
        nearest_pos = positions[np.argmin(dists)]
        direction = nearest_pos - pos
        return direction

    def obs_nearest_extractor_on_resource(pos):
        ''' Returns the direction to the nearest extractor that is on a resource in the observation. '''
        layer_extractors = observation[self.MAP_LAYERS.AGENTS_EXTRACTOR]
        layer_resources = observation[self.MAP_LAYERS.RESOURCES_EXTANT]
        positions_extractors = np.argwhere(layer_extractors > 0)
        positions_resources = np.argwhere(layer_resources > 0)
        if positions_extractors.shape[0] == 0 or positions_resources.shape[0] == 0:
            return None
        dmin = np.inf
        nearest_pos = None
        for pos_e in positions_extractors:
            for pos_r in positions_resources:
                if np.array_equal(pos_e, pos_r):
                    d = np.linalg.norm(pos_e - pos)
                    if d < dmin:
                        dmin = d
                        nearest_pos = pos_e
        if nearest_pos is None:
            return None
        direction = nearest_pos - pos
        return direction

    agent_pos, capabilities = get_agent_characteristics()

    if capabilities[CAP.CARRY]:
        # Check whether carrying anything. Get cargo from the observation.
        agent_cargo = observation[parallel_env_map_obs.MAP_LAYERS.RESOURCES_CARGO, tuple(agent_pos)]
        if agent_cargo > 0:
            # Find closest depot.
            direction = obs_nearest_depot(agent_pos)
        else:
            # Find closest extractor on resource.
            direction = obs_nearest_extractor_on_resource(agent_pos)
            if direction is None:
                # Find closest resource.
                direction = obs_nearest_resource(agent_pos)
        
    elif capabilities[CAP.EXTRACT]:
        # Find closest resource.
        direction = obs_nearest_resource(agent_pos)
        
    else:
        # Prospector behavior: wander randomly.
        # TODO: Implement actual search. Perhaps something like https://ieeexplore.ieee.org/document/6225106
        direction = np.random.randint(-1, 2, size=(2,), dtype=np.int32)
    
    if direction is None:
        direction = np.array([0, 0], dtype=np.int32)

    # Normalize direction to a single step in the Moore neighborhood.
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
        direction = np.round(direction).astype(np.int32)
    else:
        direction = np.array([0, 0], dtype=np.int32)
    moore_index = parallel_env_map_obs._velocity_to_moore_index(None, direction)
    return moore_index
