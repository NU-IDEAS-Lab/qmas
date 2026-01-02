import torch
import gymnasium.spaces as spaces
from isru_zoo.env.entity import *
from isru_zoo.env.isru_env import parallel_env_map_obs as pemo
from isru_zoo.env.isru_env import parallel_env_flat_map_obs as pefmo


def get_action_heuristic(args, action_space, observation, env_name):
    '''
    Wrapper function to select appropriate heuristic based on environment name.
    '''
    if env_name == "isru_zoo.isru_v0.parallel_env_map_obs":
        return _get_action_heuristic_pemo(args, action_space, observation)
    elif env_name == "isru_zoo.isru_v0.parallel_env_flat_map_obs":
        return _get_action_heuristic_pefmo(args, observation)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")


def _get_action_heuristic_pemo(args, action_space, observation):
    '''
    Uses a simple heuristic to determine the action based on agent type:
    - Prospectors: Move toward nearest unexplored area
    - Extractors: Move toward nearest resource
    - Haulers: Move toward nearest extractor that's on a resource
    '''

    def get_agent_characteristics():
        where_x_zero = observation[pemo.MAP_LAYERS.RELATIVE_POS_X] == 0
        where_y_zero = observation[pemo.MAP_LAYERS.RELATIVE_POS_Y] == 0
        agent_pos_mask = where_x_zero & where_y_zero
        agent_pos = torch.argwhere(agent_pos_mask).reshape(-1, 2)
        if agent_pos.shape[0] != 1:
            raise ValueError("Could not determine unique agent position.")
        agent_pos = agent_pos[0]

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
            raise ValueError("Could not determine agent capabilities.")

        return agent_pos, capabilities, stacked

    def obs_nearest_resource(pos):
        resources = observation[pemo.MAP_LAYERS.RESOURCES_EXTANT]
        extractors = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR]
        rel_x = observation[pemo.MAP_LAYERS.RELATIVE_POS_X]
        rel_y = observation[pemo.MAP_LAYERS.RELATIVE_POS_Y]

        mask_self = (rel_x == 0) & (rel_y == 0)
        mask_extractors = (extractors > 0.5) & ~mask_self
        mask = (resources > 0.5) & ~mask_extractors

        positions = torch.argwhere(mask).float()
        if positions.numel() == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), dim=1)
        return positions[torch.argmin(dists)] - pos

    def obs_nearest_depot(pos):
        layer = observation[pemo.MAP_LAYERS.DEPOTS]
        positions = torch.argwhere(layer > 0.5).float()
        if positions.numel() == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), dim=1)
        return positions[torch.argmin(dists)] - pos

    def obs_nearest_extractor_on_resource(pos):
        layer_e = observation[pemo.MAP_LAYERS.AGENTS_EXTRACTOR]
        layer_r = observation[pemo.MAP_LAYERS.RESOURCES_EXTANT]
        mask = (layer_e > 0.5) & (layer_r > 0.5)
        positions = torch.argwhere(mask).float()
        if positions.numel() == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), dim=1)
        return positions[torch.argmin(dists)] - pos

    def obs_nearest_unexplored(pos):
        explored = observation[pemo.MAP_LAYERS.MASK_RESOURCES_OBSERVED] > 0.5
        positions = torch.argwhere(~explored).float()
        if positions.numel() == 0:
            return None
        dists = torch.linalg.norm(positions - pos.float(), dim=1)
        return positions[torch.argmin(dists)] - pos

    agent_pos, capabilities, stacked = get_agent_characteristics()

    # Is agent sitting on a resource?
    on_resource = (
        observation[pemo.MAP_LAYERS.RESOURCES_EXTANT,
                    agent_pos[0], agent_pos[1]] > 0.5
    )

    # ----- movement heuristic -----
    if capabilities[CAP.CARRY]:
        cargo = observation[pemo.MAP_LAYERS.RESOURCES_CARGO, agent_pos[0], agent_pos[1]]
        direction = obs_nearest_depot(agent_pos) if cargo > 0 else obs_nearest_extractor_on_resource(agent_pos)
        if direction is None:
            direction = obs_nearest_resource(agent_pos)

    elif capabilities[CAP.EXTRACT]:
        direction = obs_nearest_resource(agent_pos)

    else:
        direction = obs_nearest_unexplored(agent_pos)

    if direction is None:
        direction = torch.tensor([0, 0], dtype=torch.int32)

    if stacked:
        direction = torch.randint(-1, 2, (2,))

    direction = direction.float()
    if torch.linalg.norm(direction) > 0:
        direction = torch.round(direction / torch.linalg.norm(direction)).to(torch.int32)
    else:
        direction = torch.tensor([0, 0], dtype=torch.int32)

    moore_index = pemo._velocity_to_moore_index(None, direction)
    action = action_space.sample()
    action["movement"] = moore_index

    # ----- communication heuristic -----
    communication_mode = getattr(args, "communication_mode", "nearest")
    uncertainty_layer = observation[pemo.MAP_LAYERS.UNCERTAINTY]

    request_flag = 0

    if communication_mode == "broadcast":
        mean_uncertainty = float(torch.mean(uncertainty_layer))
        if mean_uncertainty > args.broadcast_uncertainty_threshold:
            request_flag = 1

    elif communication_mode == "nearest":
        H, W = uncertainty_layer.shape
        region_size = int(args.nearest_region_size)
        threshold = args.nearest_uncertainty_threshold

        best_cell, best_val = None, float("-inf")

        for i in range(0, H, region_size):
            for j in range(0, W, region_size):
                sub = uncertainty_layer[i:i + region_size, j:j + region_size]
                if sub.numel() == 0:
                    continue
                mean_val = float(torch.mean(sub))
                if mean_val > threshold and mean_val > best_val:
                    best_val = mean_val
                    best_cell = (i + sub.shape[0] // 2, j + sub.shape[1] // 2)

        if best_cell is not None:
            request_flag = 1
            bi, bj = best_cell
            if "relative_position" in action["communication"]:
                rx = float(observation[pemo.MAP_LAYERS.RELATIVE_POS_X, bi, bj])
                ry = float(observation[pemo.MAP_LAYERS.RELATIVE_POS_Y, bi, bj])
                action["communication"]["relative_position"] = [rx, ry]

    else:
        raise ValueError(f"Unknown communication mode: {communication_mode}")

    # ----- NEW RULE -----
    # Extractors do NOT request communication while extracting
    if capabilities.get(CAP.EXTRACT, False) and on_resource:
        request_flag = 0
        if "relative_position" in action["communication"]:
            action["communication"]["relative_position"] = [0.0, 0.0]

    action["communication"]["request"][0] = request_flag
    action_flat = torch.from_numpy(spaces.flatten(action_space, action))
    return action_flat


# ===== flat obs version =====

import numpy as np


def _unit_moore(vec2: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(vec2, dtype=torch.float32)
    n = torch.linalg.norm(v)
    if n == 0 or torch.isnan(n):
        return torch.tensor([0, 0], dtype=torch.int32)
    return torch.round(torch.clamp(v / n, -1.0, 1.0)).to(torch.int32)


def _moore_index_from_step(step_ij: torch.Tensor) -> int:
    dx, dy = int(step_ij[0]), int(step_ij[1])
    return (dx + 1) * 3 + (dy + 1)


def _flat_offsets(args):
    W = int(args.world_size)
    N = W * W
    K = int(args.num_agents) - 1

    sizes = {
        "agent_position": 2,
        "agent_role": 1,
        "map": N,
        "other_agents_relative": 2 * K,
        "target_relative": 2,
        "uncertainty_map": N,
    }

    order = list(sizes.keys())
    offs, start = {}, 0
    for k in order:
        offs[k] = (start, start + sizes[k])
        start += sizes[k]
    offs["_total"] = start
    return offs


def slice_target_relative_from_flat(flat_obs, args) -> torch.Tensor:
    flat = torch.as_tensor(flat_obs, dtype=torch.float32).flatten()
    s, e = _flat_offsets(args)["target_relative"]
    return flat[s:e]


def _get_action_heuristic_pefmo(args, flat_obs) -> int:
    target_rel = slice_target_relative_from_flat(flat_obs, args)
    step = _unit_moore(target_rel)
    return _moore_index_from_step(step)