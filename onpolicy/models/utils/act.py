from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn
import numpy as np

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()

        def get_action_out(action_space):
            if action_space.__class__.__name__ in ["Tuple", "Dict"]:
                action_outs = []
                for aspace in action_space.spaces.values():
                    action_outs.extend(get_action_out(aspace))
                return action_outs

            elif action_space.__class__.__name__ == "Discrete":
                action_dim = action_space.n
                ao = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
                ao.action_dim = 1
                ao.available_action_dim = action_space.n
                return [ao]
            
            elif action_space.__class__.__name__ == "Box" and np.issubdtype(action_space.dtype, np.integer):
                action_dim = np.prod(action_space.high - action_space.low) + 1
                ao = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
                ao.action_dim = 1
                ao.available_action_dim = np.prod(action_space.high - action_space.low) + 1
                return [ao]
            
            elif action_space.__class__.__name__ == "MultiDiscrete":
                action_outs = []
                for n in action_space.nvec:
                    ao = Categorical(inputs_dim, n, use_orthogonal, gain)
                    ao.action_dim = 1
                    ao.available_action_dim = n
                    action_outs.append(ao)
                return action_outs

            elif action_space.__class__.__name__ == "Box":
                ao = DiagGaussian(inputs_dim, action_space.shape[0], use_orthogonal, gain)
                ao.action_dim = action_space.shape[0]
                ao.available_action_dim = np.prod(action_space.high - action_space.low) + 1
                return [ao]

            elif action_space.__class__.__name__ == "MultiBinary":
                ao = Bernoulli(inputs_dim, action_space.shape[0], use_orthogonal, gain)
                ao.action_dim = 1
                ao.available_action_dim = 2
                return [ao]

            else:
                raise NotImplementedError(f"Action space {action_space} not supported.")
        
        action_outs = get_action_out(action_space)
        self.action_outs = nn.ModuleList(action_outs)
    

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        actions = []
        action_log_probs = []

        available_actions_idx = 0
        for module in self.action_outs:
            if isinstance(module, DiagGaussian):
                action_logit = module(x)
            else:
                aa = available_actions[:, available_actions_idx:available_actions_idx + module.available_action_dim]
                available_actions_idx += module.available_action_dim
                action_logit = module(x, aa)
            action = action_logit.mode() if deterministic else action_logit.sample()
            action_log_prob = action_logit.log_probs(action)
            actions.append(action)
            action_log_probs.append(action_log_prob)
        
        actions = torch.cat(actions, -1)
        # action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        action_log_probs = torch.cat(action_log_probs, -1)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        action_probs = []

        available_actions_idx = 0
        for module in self.action_outs:
            if isinstance(module, DiagGaussian):
                action_logit = module(x)
            else:
                aa = available_actions[:, available_actions_idx:available_actions_idx + module.available_action_dim]
                available_actions_idx += module.available_action_dim
                action_logit = module(x, aa)
            action_prob = action_logit.probs
            action_probs.append(action_prob)
        
        action_probs = torch.cat(action_probs, -1)
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs = []
        dist_entropy = 0
        action_idx = 0
        available_actions_idx = 0

        for module in self.action_outs:
            if isinstance(module, DiagGaussian):
                action_logits = module(x)
            else:
                aa = available_actions[:, available_actions_idx:available_actions_idx + module.available_action_dim]
                available_actions_idx += module.available_action_dim
                action_logits = module(x, aa)
            a = action[:, action_idx:action_idx + module.action_dim]
            action_idx += module.action_dim
            action_log_probs.append(action_logits.log_probs(a))
            if active_masks is not None:
                dist_entropy += (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy += action_logits.entropy().mean()
        
        action_log_probs = torch.cat(action_log_probs, -1)

        return action_log_probs, dist_entropy
