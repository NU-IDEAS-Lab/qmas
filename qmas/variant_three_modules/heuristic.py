
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

        # Only consider resources that are not occupied by extractors.
        resources = observation[pemo.MAP_LAYERS.RESOURCES_EXTANT]
        extractors = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR]

        # Don't mask the agent's own position, since it might be on a resource.
        rel_pos_x = observation[pemo.MAP_LAYERS.RELATIVE_POS_X]
        rel_pos_y = observation[pemo.MAP_LAYERS.RELATIVE_POS_Y]

        # Create mask.
        mask_self = (rel_pos_x == 0) & (rel_pos_y == 0)
        mask_extractors = (extractors > 0) & ~mask_self
        mask = (resources > 0) & ~mask_extractors

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

    def obs_nearest_unexplored(pos):
        ''' Returns the direction to the nearest unexplored area in the observation. '''
        explored = observation[pemo.MAP_LAYERS.MASK_RESOURCES_OBSERVED] > 0
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
    Uses a simple heuristic to determine the movement action based on agent type:
    - Prospectors: Move toward nearest unexplored area
    - Extractors: Move toward nearest resource
    - Haulers: Move toward nearest extractor on a resource
    
    Only considers information within the agent's visible area.
    
    Args:
        observation: The agent's observation dict from flat_map_obs environment
        
    Returns:
        int: Movement action (0-8) corresponding to Moore neighborhood
    '''
    # Constants for map values
    MAP_VALUE_OBSTACLE = -2.0
    MAP_VALUE_RESOURCE = 2.0
    MAP_VALUE_UNEXPLORED = 0.0
    MAP_VALUE_EXPLORED = -1.0
    
    # Get the agent's role from the observation
    agent_role = int(observation["agent_role"][0] / 10.0)  # Convert back from the encoding
    
    # Get the world dimensions by assuming square world and taking square root of map size
    world_size = int(np.sqrt(observation["map"].shape[0]))
    world_dims = (world_size, world_size)
    
    # Reshape the map to 2D
    map_2d = observation["map"].reshape(world_dims)
    
    # Agent is always at (0,0) in its own relative coordinate system
    agent_pos = np.zeros(2, dtype=np.int32)
    
    # Function to convert relative direction to Moore neighborhood index
    def direction_to_moore_index(direction):
        # Normalize to single step in Moore neighborhood
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            direction = np.round(direction).astype(np.int32)
        else:
            direction = np.array([0, 0], dtype=np.int32)
        
        # Convert to Moore index (0-8)
        dx, dy = direction
        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        return (dx + 1) * 3 + (dy + 1)
    
    # Determine target position based on agent role
    if agent_role == AGENT_ROLE.HAULER.value:
        # Check if we have a target in the target_relative field
        if np.any(observation["target_relative"] != 0) and np.all(np.isfinite(observation["target_relative"])):
            return direction_to_moore_index(observation["target_relative"])
        
        # Look for extractors in other_agents_relative
        # We want extractors that might be on resources
        for i in range(observation["other_agents_relative"].shape[0]):
            # Skip agents that aren't visible
            if not np.all(np.isfinite(observation["other_agents_relative"][i])):
                continue
                
            # Get the absolute position of this agent in our map
            other_agent_rel_pos = observation["other_agents_relative"][i]
            other_agent_map_x = int(world_size/2 + other_agent_rel_pos[0])
            other_agent_map_y = int(world_size/2 + other_agent_rel_pos[1])
            
            # Check if this position is in bounds
            if (0 <= other_agent_map_x < world_size and 
                0 <= other_agent_map_y < world_size):
                
                # Check if there's a resource at this position
                map_idx = other_agent_map_x * world_size + other_agent_map_y
                if map_idx < len(observation["map"]) and observation["map"][map_idx] == MAP_VALUE_RESOURCE:
                    return direction_to_moore_index(other_agent_rel_pos)
        
        # If no extractors on resources found, move toward any visible resource
        resource_positions = np.argwhere(map_2d == MAP_VALUE_RESOURCE)
        if len(resource_positions) > 0:
            # Convert positions to relative coordinates
            relative_positions = resource_positions - np.array([world_size/2, world_size/2])
            distances = np.linalg.norm(relative_positions, axis=1)
            nearest_idx = np.argmin(distances)
            return direction_to_moore_index(relative_positions[nearest_idx])
            
    elif agent_role == AGENT_ROLE.EXTRACTOR.value:
        # Move toward nearest resource
        resource_positions = np.argwhere(map_2d == MAP_VALUE_RESOURCE)
        if len(resource_positions) > 0:
            # Convert positions to relative coordinates
            relative_positions = resource_positions - np.array([world_size/2, world_size/2])
            distances = np.linalg.norm(relative_positions, axis=1)
            nearest_idx = np.argmin(distances)
            return direction_to_moore_index(relative_positions[nearest_idx])
            
    elif agent_role == AGENT_ROLE.PROSPECTOR.value:
        # Move toward nearest unexplored area
        unexplored_positions = np.argwhere(map_2d == MAP_VALUE_UNEXPLORED)
        if len(unexplored_positions) > 0:
            # Convert positions to relative coordinates
            relative_positions = unexplored_positions - np.array([world_size/2, world_size/2])
            distances = np.linalg.norm(relative_positions, axis=1)
            nearest_idx = np.argmin(distances)
            return direction_to_moore_index(relative_positions[nearest_idx])
    
    # If we haven't returned by now, use a random movement
    random_direction = np.random.randint(-1, 2, 2)
    return direction_to_moore_index(random_direction)
    