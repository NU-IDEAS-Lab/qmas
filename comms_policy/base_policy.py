import torch
import torch.nn as nn

class CommunicationBaseModule(nn.Module):
    """
    Communication network class for QMAS. Outputs communications given global state estimate.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param share_obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, share_obs_space, action_space, device=torch.device("cpu")):
        raise NotImplementedError("CommunicationBaseModule is an abstract class and should not be instantiated.")


    def forward(self, state, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param share_obs: (np.ndarray / torch.Tensor) global state estimate into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros. Used for RNNs.
        :param available_actions: (np.ndarray / torch.Tensor) action mask that denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        raise NotImplementedError("CommunicationBaseModule is an abstract class and should not be instantiated.")
    

    def evaluate_actions(self, state, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        raise NotImplementedError("CommunicationBaseModule is an abstract class and should not be instantiated.")