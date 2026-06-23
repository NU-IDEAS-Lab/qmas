
import torch
from isru_zoo.env.entity import *
from isru_zoo.env.isru_env import parallel_env_map_obs as pemo
from isru_zoo.env.isru_env import parallel_env_flat_map_obs as pefmo

def get_movement_action_heuristic(args,observation, env_name):
    '''
    Wrapper function to select appropriate heuristic based on environment name.
    '''
    if env_name == "isru_zoo.isru_v0.parallel_env_map_obs":
        return _get_movement_action_heuristic_pemo(observation)
    elif env_name == "isru_zoo.isru_v0.parallel_env_flat_map_obs":
        return _get_movement_action_heuristic_pefmo(args,observation)
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
        if observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR_EXTRACTOR, agent_pos[0], agent_pos[1]] >= 2.0:
            capabilities = {CAP.EXTRACT: True, CAP.CARRY: False, CAP.PROSPECT: True}
            stacked = observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR_EXTRACTOR, agent_pos[0], agent_pos[1]] > 2.0
        elif observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR, agent_pos[0], agent_pos[1]] >= 2.0:
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
        extractors = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR] + observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR_EXTRACTOR]

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
        layer_extractors = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR] + observation[pemo.MAP_LAYERS.AGENTS_PROSPECTOR_EXTRACTOR]
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

import torch
import numpy as np

def _unit_moore(vec2: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(vec2, dtype=torch.float32)
    n = torch.linalg.norm(v)
    if n == 0 or torch.isnan(n):
        return torch.tensor([0, 0], dtype=torch.int32)
    v = torch.clamp(v / n, -1.0, 1.0)
    return torch.round(v).to(torch.int32)  # [dx, dy] in {-1,0,1}

def _moore_index_from_step(step_ij: torch.Tensor) -> int:
    dx, dy = int(step_ij[0].item()), int(step_ij[1].item())
    return (dx + 1) * 3 + (dy + 1)  # 0..8

def _flat_offsets(args):
    """
    Compute [start, end) offsets for the top-level fields in your flat layout,
    using only args.world_size and args.num_agents.
    """
    W = int(getattr(args, "world_size"))
    N = W * W
    K = int(getattr(args, "num_agents")) - 1  # other agents

    sizes = {
        "agent_position": 2,
        "agent_role": 1,
        "map": N,
        "other_agents_relative": 2 * K,
        "target_relative": 2,
        "uncertainty_map": N,
    }
    # cumulative offsets in the fixed order
    order = [
        "agent_position",
        "agent_role",
        "map",
        "other_agents_relative",
        "target_relative",
        "uncertainty_map",
    ]
    offs = {}
    start = 0
    for key in order:
        size = sizes[key]
        offs[key] = (start, start + size)
        start += size
    offs["_total"] = start
    return offs

def slice_target_relative_from_flat(flat_obs, args) -> torch.Tensor:
    flat = torch.as_tensor(flat_obs, dtype=torch.float32).flatten()
    offs = _flat_offsets(args)
    s, e = offs["target_relative"]
    if flat.numel() < e:
        raise ValueError(f"flat_obs too short (have {flat.numel()}, need at least {e})")
    return flat[s:e]  # shape (2,)

def _get_movement_action_heuristic_pefmo(args,flat_obs) -> int:
    """
    1) Slice target_relative using args.world_size & args.num_agents
    2) Convert to one-cell Moore step
    3) Return discrete action index (0..8)
    """
    target_rel = slice_target_relative_from_flat(flat_obs, args)   # (2,)
    step = _unit_moore(target_rel)                                 # int32 [dx, dy]
    return _moore_index_from_step(step)
