
import torch
from isru_zoo.env.entity import *
from isru_zoo.env.isru_env import parallel_env_map_obs as pemo
from isru_zoo.env.isru_env import parallel_env_flat_map_obs as pefmo

def get_movement_action_heuristic(observation, obs_class=pemo):
    if obs_class == pemo:
        return _get_movement_action_heuristic_pemo(observation)
    elif obs_class == pefmo:
        return _get_movement_action_heuristic_pefmo(observation)
    else:
        raise ValueError(f"Unsupported observation class {obs_class} for heuristic.")

def _get_movement_action_heuristic_pemo(observation):
    ''' Uses a simple heuristic to determine the action for the given agent based on the observation. '''

    # Just move to the bottom-right corner for now as an example.
    # return pemo._velocity_to_moore_index(None, torch.tensor([1, 1]))

    def get_agent_characteristics():
        ''' Gets agent position and capabilities from the observation. '''

        # Position is where both relative pos x and y are zero.
        where_x_zero = observation[pemo.MAP_LAYERS.RELATIVE_POS_X] == 0
        where_y_zero = observation[pemo.MAP_LAYERS.RELATIVE_POS_Y] == 0
        agent_pos = torch.argwhere(where_x_zero & where_y_zero).reshape(-1, 2)
        if agent_pos.shape[0] != 1:
            raise ValueError("Could not determine unique agent position from observation.")
        agent_pos = agent_pos[0]

        # Determine role based on agent layers.
        stacked = False
        if observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] >= 255.0:
            capabilities = {CAP.EXTRACT: True, CAP.CARRY: False, CAP.PROSPECT: False}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] > 255.0
        elif observation[pemo.MAP_LAYERS.AGENTS_HAULER, agent_pos[0], agent_pos[1]] >= 255.0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: True, CAP.PROSPECT: False}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_HAULER, agent_pos[0], agent_pos[1]] > 255.0
        elif observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR, agent_pos[0], agent_pos[1]] >= 255.0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: False, CAP.PROSPECT: True}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR, agent_pos[0], agent_pos[1]] > 255.0
        else:
            raise ValueError("Could not determine agent capabilities from observation.")

        return agent_pos, capabilities, stacked

    def obs_nearest_resource(pos):
        ''' Returns the direction to the nearest resource in the observation. '''
        layer = observation[pemo.MAP_LAYERS.RESOURCES_EXTANT]
        positions = torch.argwhere(layer > 0).float()
        if positions.shape[0] == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), axis=1)
        nearest_pos = positions[torch.argmin(dists)]
        direction = nearest_pos - pos
        return direction

    def obs_nearest_depot(pos):
        ''' Returns the direction to the nearest depot in the observation. '''
        layer = observation[pemo.MAP_LAYERS.DEPOTS]
        positions = torch.argwhere(layer > 0).float()
        if positions.shape[0] == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), axis=1)
        nearest_pos = positions[torch.argmin(dists)]
        direction = nearest_pos - pos
        return direction

    def obs_nearest_extractor_on_resource(pos):
        ''' Returns the direction to the nearest extractor that is on a resource in the observation. '''
        layer_extractors = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR]
        layer_resources = observation[pemo.MAP_LAYERS.RESOURCES_EXTANT]
        positions_extractors = torch.argwhere(layer_extractors > 0)
        positions_resources = torch.argwhere(layer_resources > 0)
        if positions_extractors.shape[0] == 0 or positions_resources.shape[0] == 0:
            return None
        dmin = torch.inf
        nearest_pos = None
        for pos_e in positions_extractors:
            for pos_r in positions_resources:
                if torch.all(pos_e == pos_r):
                    d = torch.linalg.norm(pos_e.float() - pos.float())
                    if d < dmin:
                        dmin = d
                        nearest_pos = pos_e.float()
        if nearest_pos is None:
            return None
        direction = nearest_pos - pos
        return direction

    agent_pos, capabilities, stacked = get_agent_characteristics()

    if capabilities[CAP.CARRY]:
        # Check whether carrying anything. Get cargo from the observation.
        agent_cargo = observation[pemo.MAP_LAYERS.RESOURCES_CARGO, agent_pos[0], agent_pos[1]]
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
        direction = torch.randint(-1, 2, (2,))
    
    if direction is None:
        direction = torch.tensor([0, 0], dtype=torch.int32)
    
    # Handle case where two of same agent type are stacked.
    if stacked:
        direction = torch.randint(-1, 2, (2,))

    # Normalize direction to a single step in the Moore neighborhood.
    direction = direction.float()
    if torch.linalg.norm(direction) > 0:
        direction = direction / torch.linalg.norm(direction)
        direction = torch.round(direction).to(torch.int32)
    else:
        direction = torch.tensor([0, 0], dtype=torch.int32)
    moore_index = pemo._velocity_to_moore_index(None, direction)
    return moore_index

def _get_movement_action_heuristic_pefmo(observation):
    ''' Uses a simple heuristic to determine the action for the given agent based on the observation. '''

    # Just move to the bottom-right corner for now as an example.
    # return pemo._velocity_to_moore_index(None, torch.tensor([1, 1]))
    action_none = pefmo._velocity_to_moore_index(None, torch.tensor([0, 0]))

    def get_agent_characteristics():
        ''' Gets agent position and capabilities from the observation. '''

        agent_pos = observation["agent_position"]
        agent_pos = torch.round(agent_pos).to(torch.int32)
        if agent_pos.shape[0] != 1:
            raise ValueError("Could not determine unique agent position from observation.")
        agent_pos = agent_pos[0]

        # Determine role based on agent layers.
        stacked = False
        if observation[pef.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] >= 255.0:
            capabilities = {CAP.EXTRACT: True, CAP.CARRY: False, CAP.PROSPECT: False}
            stacked = observation[pef.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] > 255.0
        elif observation[pef.MAP_LAYERS.AGENTS_HAULER, agent_pos[0], agent_pos[1]] >= 255.0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: True, CAP.PROSPECT: False}
            stacked = observation[pef.MAP_LAYERS.AGENTS_HAULER, agent_pos[0], agent_pos[1]] > 255.0
        elif observation[pef.MAP_LAYERS.AGENTS_PROSPECTOR, agent_pos[0], agent_pos[1]] >= 255.0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: False, CAP.PROSPECT: True}
            stacked = observation[pef.MAP_LAYERS.AGENTS_PROSPECTOR, agent_pos[0], agent_pos[1]] > 255.0
        else:
            raise ValueError("Could not determine agent capabilities from observation.")