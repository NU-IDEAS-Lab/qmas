
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
    Uses a simple heuristic to determine the movement action based on agent type
    for the flat map observation representation.
    
    Args:
        observation: The agent's flat observation dictionary
        
    Returns:
        int: Movement action (0-8) corresponding to Moore neighborhood
    '''
    # Extract agent role to determine agent type
    agent_role = int(observation["agent_role"][0] / 10.0)  # Role is stored as role * 10.0
    
    # Use constants from the parallel_env_flat_map_obs class
    MAP_VALUE_OBSTACLE = pefmo.MAP_VALUE_OBSTACLE
    MAP_VALUE_RESOURCE = pefmo.MAP_VALUE_RESOURCE
    MAP_VALUE_UNEXPLORED = pefmo.MAP_VALUE_UNEXPLORED
    MAP_VALUE_EXPLORED = pefmo.MAP_VALUE_EXPLORED
    
    # Reshape the map to 2D
    world_size = int(np.sqrt(len(observation["map"])))
    flat_map = observation["map"].reshape((world_size, world_size))
    
    # Get agent position (already provided in the observation)
    self_pos = observation["agent_position"].astype(np.int32)
    
    # Determine behavior based on agent type
    # PROSPECTOR = 0, EXTRACTOR = 1, HAULER = 2 in the AGENT_ROLE enum
    is_prospector = agent_role == 0
    is_extractor = agent_role == 1
    is_hauler = agent_role == 2
    
    # Calculate target based on agent type
    if is_prospector:
        # For prospector: find nearest unexplored area
        unexplored = flat_map == MAP_VALUE_UNEXPLORED
        if np.any(unexplored):
            unexplored_positions = np.where(unexplored)
            distances = np.sqrt((unexplored_positions[0] - self_pos[0])**2 + 
                             (unexplored_positions[1] - self_pos[1])**2)
            nearest_idx = np.argmin(distances)
            target_x, target_y = unexplored_positions[0][nearest_idx], unexplored_positions[1][nearest_idx]
        else:
            # If everything is explored, use target from observation if available
            if not np.all(observation["target_relative"] == 0):
                target_x = int(self_pos[0] + observation["target_relative"][0])
                target_y = int(self_pos[1] + observation["target_relative"][1])
            else:
                # If no target, move randomly
                return np.random.randint(0, 9)
            
    elif is_extractor:
        # For extractor: find nearest resource
        resources = flat_map == MAP_VALUE_RESOURCE
        if np.any(resources):
            resource_positions = np.where(resources)
            distances = np.sqrt((resource_positions[0] - self_pos[0])**2 + 
                             (resource_positions[1] - self_pos[1])**2)
            nearest_idx = np.argmin(distances)
            target_x, target_y = resource_positions[0][nearest_idx], resource_positions[1][nearest_idx]
        else:
            # If no resources visible, use target from observation if available
            if not np.all(observation["target_relative"] == 0):
                target_x = int(self_pos[0] + observation["target_relative"][0])
                target_y = int(self_pos[1] + observation["target_relative"][1])
            else:
                # If no target, move randomly
                return np.random.randint(0, 9)
            
    elif is_hauler:
        # For hauler, we don't have direct info on extractors' positions on resources
        # in the flat map, so we use target_relative from the observation
        if not np.all(observation["target_relative"] == 0):
            target_x = int(self_pos[0] + observation["target_relative"][0])
            target_y = int(self_pos[1] + observation["target_relative"][1])
        else:
            # Check if other agents (likely extractors) are in position
            other_agents = observation["other_agents_relative"]
            if len(other_agents) > 0:
                # Find nearest other agent
                distances = np.sqrt(np.sum(other_agents**2, axis=1))
                nearest_idx = np.argmin(distances)
                target_x = int(self_pos[0] + other_agents[nearest_idx][0])
                target_y = int(self_pos[1] + other_agents[nearest_idx][1])
            else:
                # If no target or other agents, move randomly
                return np.random.randint(0, 9)
    
    # Calculate direction to target
    dx = np.sign(target_x - self_pos[0])
    dy = np.sign(target_y - self_pos[1])
    
    # Check if the move would hit an obstacle
    next_x, next_y = self_pos[0] + dx, self_pos[1] + dy
    if (0 <= next_x < world_size and 
        0 <= next_y < world_size and 
        flat_map[next_x, next_y] == MAP_VALUE_OBSTACLE):
        # If obstacle in direct path, try to move around it
        # Try alternative directions (prioritize maintaining one component of direction)
        alternatives = []
        if dx != 0:
            alternatives.append((dx, 0))  # Keep x-direction
        if dy != 0:
            alternatives.append((0, dy))  # Keep y-direction
        if dx != 0 and dy != 0:
            alternatives.append((dx, -dy))  # Diagonal alternative 1
            alternatives.append((-dx, dy))  # Diagonal alternative 2
        
        # Try each alternative
        for alt_dx, alt_dy in alternatives:
            alt_x, alt_y = self_pos[0] + alt_dx, self_pos[1] + alt_dy
            if (0 <= alt_x < world_size and 
                0 <= alt_y < world_size and 
                flat_map[alt_x, alt_y] != MAP_VALUE_OBSTACLE):
                dx, dy = alt_dx, alt_dy
                break
    
    # Convert to Moore neighborhood index (0-8)
    # [0 1 2]
    # [3 4 5]
    # [6 7 8]
    moore_idx = (dx + 1) * 3 + (dy + 1)
    
    return moore_idx
