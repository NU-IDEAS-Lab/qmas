
import torch
import gymnasium.spaces as spaces
from qmas.variant_two_modules.policy import QmasPolicy as QmasPolicyBase
from .heuristic import get_movement_action_heuristic


class QmasPolicy(QmasPolicyBase):
    ''' This class implements the QMAS policy, using heuristic for movement and parent for communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        # Assume act_space is a dict with keys 'movement' and 'communication'
        comm_act_space = act_space['communication']
        # Only pass the communication action space to the actor
        super().__init__(args, obs_space, cent_obs_space, comm_act_space, device)
        self.full_act_space = act_space

        self.env_class = args.env_class

        # Use the flatten function to determine which indices of actions correspond to communication vs movement.
        action = self.full_act_space.sample()
        DUMMY_VALUE = 123456789.0 # Unique value to identify communication part.
        action["communication"]["request"] = torch.tensor([DUMMY_VALUE])
        action["communication"]["relative_position"] = torch.tensor([DUMMY_VALUE, DUMMY_VALUE]) 
        action_flat = torch.from_numpy(spaces.flatten(self.full_act_space, action)).float()
        self.action_comm_mask = torch.isclose(action_flat, torch.tensor(DUMMY_VALUE, dtype=torch.float32))
        self.action_comm_indices = torch.argwhere(self.action_comm_mask).squeeze(-1)
        self.action_move_indices = torch.argwhere(~self.action_comm_mask).squeeze(-1)


    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        Compute actions: movement from heuristic, communication from parent.
        agents: list of agent objects (needed for heuristic)
        raw_obs: list of raw observation dicts (needed for heuristic)
        """
        # Get communication actions from parent
        values, comm_actions, comm_log_probs, rnn_states_actor, rnn_states_critic = super().get_actions(
            cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions, deterministic)

        # Get movement actions from heuristic
        movement_actions = torch.zeros((obs.shape[0], self.action_move_indices.shape[0]), dtype=comm_actions.dtype)
        for i in range(obs.shape[0]):
            action = get_movement_action_heuristic(obs[i], self.env_class)
            movement_actions[i] = action

        # Combine into full action dict
        actions = torch.zeros((comm_actions.shape[0], spaces.flatdim(self.full_act_space)), dtype=comm_actions.dtype, device=comm_actions.device)
        actions[:, self.action_comm_indices] = comm_actions
        actions[:, self.action_move_indices] = movement_actions.to(comm_actions.device)

        return values, actions, comm_log_probs, rnn_states_actor, rnn_states_critic


    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate only the communication part of the action, ignoring movement.
        Expects action to be a list/dict with 'communication' and 'movement' keys.
        """
        
        # Extract only the communication part using our precomputed indices.
        action_comms = action[:, self.action_comm_indices]

        # Call parent evaluate_actions with only communication actions
        return super().evaluate_actions(
            cent_obs, obs, rnn_states_actor, rnn_states_critic, action_comms, masks, available_actions, active_masks
        )

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        Movement actions are computed using a heuristic, communication actions from the parent class.
        :param obs: (torch.Tensor) input to network.
        :param rnn_states_actor: (torch.Tensor) if actor is RNN, RNN states for actor.
        :param masks: (torch.Tensor) denotes points at which RNN states should be reset.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from the action distribution or use the mode.

        :return actions: (torch.Tensor) actions to take.
        :return rnn_states_actor: (torch.Tensor) updated RNN states for actor.
        """
        # Get communication actions from parent
        comm_actions, rnn_states_actor = super().act(
            obs, rnn_states_actor, masks, available_actions, deterministic)

        # Get movement actions from heuristic
        movement_actions = torch.zeros((obs.shape[0], self.action_move_indices.shape[0]), dtype=comm_actions.dtype)
        for i in range(obs.shape[0]):
            action = get_movement_action_heuristic(obs[i], self.env_class)
            movement_actions[i] = action

        # Combine into full action dict
        actions = torch.zeros((comm_actions.shape[0], spaces.flatdim(self.full_act_space)), dtype=comm_actions.dtype, device=comm_actions.device)
        actions[:, self.action_comm_indices] = comm_actions
        actions[:, self.action_move_indices] = movement_actions.to(comm_actions.device)

        return actions, rnn_states_actor