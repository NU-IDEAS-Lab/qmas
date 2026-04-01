from collections import deque
import torch

class TrajectoryBuffer:
    ''' A simple class to hold trajectory data. '''

    def __init__(self, history_length, transition_includes_actions=False):
        self.buffer = deque(maxlen=history_length)
        self.transition_includes_actions = transition_includes_actions
        self.history_length = history_length

    def reset(self):
        ''' Clear the buffer. '''
        self.buffer.clear()

    def add(self, obs, visibility_mask, action=None):
        ''' Add a new transition to the buffer. '''

        if self.transition_includes_actions and action is not None:
            transition = torch.cat([action, obs], dim=-1)
            # Pad visibility_mask with ones for the action dimensions so it matches the transition shape.
            action_mask = torch.ones(*action.shape[:-1], action.shape[-1], dtype=visibility_mask.dtype, device=visibility_mask.device)
            visibility_mask = torch.cat([action_mask, visibility_mask], dim=-1)
        else:
            transition = obs
        
        self.buffer.append((transition, visibility_mask))

    def ready(self):
        ''' Check if the buffer has enough data to be used for prediction. '''
        return len(self.buffer) == self.history_length

    def get_trajectory(self):
        ''' Get the current trajectory as a tensor. '''
        if len(self.buffer) == 0:
            return None, None
        transitions, visibility_masks = zip(*self.buffer)
        trajectory = torch.stack(transitions, dim=0)
        visibility_mask = torch.stack(visibility_masks, dim=0)
        return trajectory, visibility_mask

class MultiAgentTrajectoryBuffer:
    ''' A class to hold trajectory data for multiple agents. '''

    def __init__(self, num_agents, history_length, transition_includes_actions=False):
        self.buffers = [TrajectoryBuffer(history_length, transition_includes_actions) for _ in range(num_agents)]

    def reset(self):
        ''' Reset all buffers. '''
        for buffer in self.buffers:
            buffer.reset()

    def add(self, agent_id, obs, visibility_mask, action=None):
        ''' Add a new transition for a specific agent. '''
        self.buffers[agent_id].add(obs, visibility_mask, action)

    def ready(self):
        ''' Check if all buffers are ready for prediction. '''
        return all(buffer.ready() for buffer in self.buffers)

    def get_trajectories(self):
        ''' Get the current trajectories for all agents. '''
        trajectories = []
        visibility_masks = []
        for buffer in self.buffers:
            trajectory, visibility_mask = buffer.get_trajectory()
            trajectories.append(trajectory)
            visibility_masks.append(visibility_mask)
        return trajectories, visibility_masks

    def get_trajectory(self, agent_id):
        ''' Get the current trajectory for a specific agent. '''
        return self.buffers[agent_id].get_trajectory()