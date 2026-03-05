
import torch
import gymnasium.spaces as spaces
from qmas.variant_two_modules.policy import QmasPolicy as QmasPolicyBase
from qmas.comms_heuristic.heuristic import get_action_heuristic


class QmasPolicy(QmasPolicyBase):
    ''' This class implements the QMAS policy. '''

    IS_TRAINABLE = False

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        # Just run the normal init function for now. It creates NNs which we don't use, but this is simplest.
        super().__init__(args, obs_space, cent_obs_space, act_space, device)

        self.action_dim = spaces.flatdim(act_space)


    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        Compute actions: movement and communication from heuristic.
        agents: list of agent objects (needed for heuristic)
        raw_obs: list of raw observation dicts (needed for heuristic)
        """
        
        actions, rnn_states_actor = self.act(obs, rnn_states_actor, masks, available_actions, deterministic)
        
        # Provide dummy tensors for values and log_probs since they are not used.
        values = torch.zeros((cent_obs.shape[0], 1, 1), dtype=torch.float32)
        log_probs = torch.zeros((obs.shape[0], 1), dtype=torch.float32)

        return values, actions, log_probs, rnn_states_actor, rnn_states_critic


    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate only the communication part of the action, ignoring movement.
        Expects action to be a list/dict with 'communication' and 'movement' keys.
        """
        
        raise NotImplementedError("QmasPolicy does not implement evaluate_actions since it is not trainable.")


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
        # Get actions from heuristic
        actions = torch.zeros((obs.shape[0], self.action_dim), dtype=torch.float32)
        for i in range(obs.shape[0]):
            action = get_action_heuristic(self.args, self.act_space, obs[i], self.args.env_class)
            actions[i] = action

        return actions, rnn_states_actor