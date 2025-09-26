
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
    # Extract important layers from observation
    MAP_LAYERS = pemo.MAP_LAYERS
    
    # Find self position (where relative position X and Y are both 0)
    rel_pos_x = observation[MAP_LAYERS.RELATIVE_POS_X]
    rel_pos_y = observation[MAP_LAYERS.RELATIVE_POS_Y]
    self_pos = np.where((rel_pos_x == 0) & (rel_pos_y == 0))
    self_x, self_y = self_pos[0][0], self_pos[1][0]  # Extract coordinates
    
    # Determine agent type based on which layer has a high value at self position
    agent_prospector = observation[MAP_LAYERS.AGENTS_PROSPECTOR][self_x, self_y]
    agent_extractor = observation[MAP_LAYERS.AGENTS_EXTRACTOR][self_x, self_y]
    agent_hauler = observation[MAP_LAYERS.AGENTS_HAULER][self_x, self_y]
    
    # Determine agent type
    is_prospector = agent_prospector > 250  # Using threshold to detect value of 255
    is_extractor = agent_extractor > 250
    is_hauler = agent_hauler > 250
    
    # Get observed mask and resource mask
    observed_mask = observation[MAP_LAYERS.MASK_OBSERVED]
    resources_extant = observation[MAP_LAYERS.RESOURCES_EXTANT]
    
    # Calculate target based on agent type
    if is_prospector:
        # For prospector: find nearest unexplored area
        unexplored = observed_mask < 0.5
        if torch.any(unexplored):
            unexplored_positions = np.where(unexplored)
            distances = np.sqrt((unexplored_positions[0] - self_x)**2 + 
                               (unexplored_positions[1] - self_y)**2)
            nearest_idx = np.argmin(distances)
            target_x, target_y = unexplored_positions[0][nearest_idx], unexplored_positions[1][nearest_idx]
        else:
            # If everything is explored, move randomly
            return np.random.randint(0, 9)
            
    elif is_extractor:
        # For extractor: find nearest resource
        if torch.any(resources_extant > 0):
            resource_positions = np.where(resources_extant > 0)
            distances = np.sqrt((resource_positions[0] - self_x)**2 + 
                               (resource_positions[1] - self_y)**2)
            nearest_idx = np.argmin(distances)
            target_x, target_y = resource_positions[0][nearest_idx], resource_positions[1][nearest_idx]
        else:
            # If no resources visible, move randomly
            return np.random.randint(0, 9)
            
    elif is_hauler:
        # For hauler: find nearest extractor that's on a resource
        extractor_positions = np.where(observation[MAP_LAYERS.AGENTS_EXTRACTOR] > 0)
        extractors_on_resource = []
        
        for i in range(len(extractor_positions[0])):
            x, y = extractor_positions[0][i], extractor_positions[1][i]
            if resources_extant[x, y] > 0:
                extractors_on_resource.append((x, y))
        
        if extractors_on_resource:
            # Calculate distances to extractors on resources
            distances = []
            for x, y in extractors_on_resource:
                dist = np.sqrt((x - self_x)**2 + (y - self_y)**2)
                distances.append(dist)
            # Find nearest extractor on resource
            nearest_idx = np.argmin(distances)
            target_x, target_y = extractors_on_resource[nearest_idx]
        else:
            # If no extractors on resources, find nearest resource
            if torch.any(resources_extant > 0):
                resource_positions = np.where(resources_extant > 0)
                distances = np.sqrt((resource_positions[0] - self_x)**2 + 
                                   (resource_positions[1] - self_y)**2)
                nearest_idx = np.argmin(distances)
                target_x, target_y = resource_positions[0][nearest_idx], resource_positions[1][nearest_idx]
            else:
                # If no resources visible, move randomly
                return np.random.randint(0, 9)
    
    # Calculate direction to target
    dx = np.sign(target_x - self_x)
    dy = np.sign(target_y - self_y)
    
    # Convert to Moore neighborhood index (0-8)
    # [0 1 2]
    # [3 4 5]
    # [6 7 8]
    moore_idx = (dx + 1) * 3 + (dy + 1)
    
    return moore_idx

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

