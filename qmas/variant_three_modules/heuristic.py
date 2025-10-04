
import torch
from isru_zoo.env.entity import *
from isru_zoo.env.isru_env import parallel_env_map_obs as pemo
from isru_zoo.env.isru_env import parallel_env_flat_map_obs as pefmo

def get_movement_action_heuristic(observation, env_name):
    '''
    Wrapper function to select appropriate heuristic based on environment name.
    '''
    if env_name == "isru_zoo.isru_v0.parallel_env_map_obs":
        return _get_movement_action_heuristic_pemo(observation)
    elif env_name == "isru_zoo.isru_v0.parallel_env_flat_map_obs":
        return _get_movement_action_heuristic_pefmo(observation)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

def _get_movement_action_heuristic_pemo(observation):
    '''
    Uses a simple heuristic to determine the movement action based on agent type:
    - Prospectors: Move toward nearest unexplored area
    - Extractors: Move toward nearest resource
    - Haulers: Move toward nearest extractor that's on a resource
    
    Args:
        observation: The agent's combined observation (stacked layers)
        
    Returns:
        int: Movement action (0-8) corresponding to Moore neighborhood
    '''
    def get_agent_characteristics():
        ''' Gets agent position and capabilities from the observation. '''

        # Position is where both relative pos x and y are zero.
        where_x_zero = observation[pemo.MAP_LAYERS.RELATIVE_POS_X] == 0
        where_y_zero = observation[pemo.MAP_LAYERS.RELATIVE_POS_Y] == 0
        agent_pos_mask = where_x_zero & where_y_zero
        agent_pos = torch.argwhere(agent_pos_mask).reshape(-1, 2)
        if agent_pos.shape[0] != 1:
            raise ValueError("Could not determine unique agent position from observation.")
        agent_pos = agent_pos[0]

        # Determine role based on agent layers.
        stacked = False
        if observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] >= 2.0:
            capabilities = {CAP.EXTRACT: True, CAP.CARRY: False, CAP.PROSPECT: False}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] > 2.0
        elif observation[pemo.MAP_LAYERS.AGENTS_HAULER, agent_pos[0], agent_pos[1]] >= 2.0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: True, CAP.PROSPECT: False}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_HAULER, agent_pos[0], agent_pos[1]] > 2.0
        elif observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR, agent_pos[0], agent_pos[1]] >= 2.0:
            capabilities = {CAP.EXTRACT: False, CAP.CARRY: False, CAP.PROSPECT: True}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR, agent_pos[0], agent_pos[1]] > 2.0
        else:
            raise ValueError("Could not determine agent capabilities from observation.")

        return agent_pos, capabilities, stacked

    def obs_nearest_resource(pos):
        ''' Returns the direction to the nearest resource in the observation. '''

        # Only consider resources that are not occupied by extractors.
        resources = observation[pemo.MAP_LAYERS.RESOURCES_EXTANT]
        extractors = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR]

        # Don't mask the agent's own position, since it might be on a resource.
        rel_pos_x = observation[pemo.MAP_LAYERS.RELATIVE_POS_X]
        rel_pos_y = observation[pemo.MAP_LAYERS.RELATIVE_POS_Y]

        # Create mask.
        mask_self = (rel_pos_x == 0) & (rel_pos_y == 0)
        mask_extractors = (extractors > 0.5) & ~mask_self
        mask = (resources > 0.5) & ~mask_extractors

        positions = torch.argwhere(mask).float()
        if positions.shape[0] == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), axis=1)
        nearest_pos = positions[torch.argmin(dists)]
        direction = nearest_pos - pos
        return direction

    def obs_nearest_depot(pos):
        ''' Returns the direction to the nearest depot in the observation. '''
        layer = observation[pemo.MAP_LAYERS.DEPOTS]
        positions = torch.argwhere(layer > 0.5).float()
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
        mask = (layer_extractors > 0.5) & (layer_resources > 0.5)
        positions = torch.argwhere(mask).float()
        if positions.shape[0] == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), axis=1)
        nearest_pos = positions[torch.argmin(dists)]
        direction = nearest_pos - pos
        return direction

    def obs_nearest_unexplored(pos):
        ''' Returns the direction to the nearest unexplored area in the observation. '''
        explored = observation[pemo.MAP_LAYERS.MASK_RESOURCES_OBSERVED] > 0.5
        positions = torch.argwhere(~explored).float()
        if positions.shape[0] == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), axis=1)
        nearest_pos = positions[torch.argmin(dists)]
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
        # Prospector behavior: move toward nearest unexplored area.
        direction = obs_nearest_unexplored(agent_pos)
    
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
    '''
    Simple movement heuristic for parallel_env_flat_map_obs environment.
    Uses agent role and target_relative to select a movement action.
    Args:
        observation: dict with keys 'agent_role', 'agent_position', 'target_relative', etc.
    Returns:
        int: Movement action (0-8) corresponding to Moore neighborhood
    '''
    import numpy as np
    # The observation is a flattened array. We need to extract the correct slices.
    # The structure is (see _observe):
    #   agent_role: 1
    #   agent_position: 2
    #   target_relative: 2
    #   other_agents_relative: N*2 (N = num other agents, unknown here)
    #   map: prod(world_dims)
    #   uncertainty_map: prod(world_dims)
    # We'll use only the first 5 elements (agent_role, agent_position, target_relative)
    agent_role = observation[0]
    agent_position = observation[1:3]
    # target_relative = observation[3:5]
    # If target is not visible (all zeros), do not move
    # if np.allclose(target_relative, 0):
    #     return 4  # Center (no movement)

    if agent_role == AGENT_ROLE.PROSPECTOR:
        # Move randomly for prospectors
        target_relative = np.random.randint(-1, 2, size=2)
    elif agent_role == AGENT_ROLE.EXTRACTOR:
        # Move toward nearest resource (assumed to be at target_relative)
        target_relative = observation[3:5]
        if np.allclose(target_relative, 0):
            return 4  # No movement if no target
    elif agent_role == AGENT_ROLE.HAULER:
        # Move toward nearest extractor (assumed to be at target_relative)
        target_relative = observation[3:5]
        if np.allclose(target_relative, 0):
            return 4  # No movement if no target
    else:
        # Unknown role, do not move
        return 4  # Center (no movement)

    # Compute direction to target
    dx = int(np.sign(target_relative[0]))
    dy = int(np.sign(target_relative[1]))
    # Convert to Moore neighborhood index (0-8)
    # [0 1 2]
    # [3 4 5]
    # [6 7 8]
    moore_idx = (dx + 1) * 3 + (dy + 1)
    return moore_idx

