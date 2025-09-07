import torch
import segmentation_models_pytorch as smp
import gymnasium.spaces as spaces
from onpolicy.models.r_actor_critic import R_Critic

from onpolicy.models.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space

class QmasActor(torch.nn.Module):
    ''' This class implements the QMAS actor, including communication (2-module variant). '''
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.args = args
        self.obs_space = obs_space
        self.action_space = action_space

        obs_shape = get_shape_from_obs_space(obs_space)
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        action_dim = spaces.flatdim(action_space)

        self.unet = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=input_channel,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=action_dim,                      # model output channels (number of classes in your dataset)
            # decoder_interpolation="bilinear",
            activation="sigmoid",          # activation function
        )

        self.act = ACTLayer(action_space, action_dim, self.args.use_orthogonal, self.args.gain)

        self.to(self.device)


    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param obs: (torch.Tensor) input to network.
        :param rnn_states: (torch.Tensor) if using RNN, the current rnn states.
        :param masks: (torch.Tensor) if using RNN, denotes whether rnn states should be reset.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated rnn states.
        """
        
        input_shape = obs.shape

        # Observations are batch, channel, width, height (B, C, W, H) tensors.
        x = self.unet(obs)
        
        # x is now (B, action_dim, W, H). Restructure it to (B * W * H, action_dim).
        x = x.permute(0, 2, 3, 1).view(-1, x.shape[1])

        # Get the actions and log probabilities.
        actions, action_log_probs = self.act(x, available_actions, deterministic)

        # Restore the actions and log probabilities to (B, W, H, action_dim).
        actions = actions.view(input_shape[0], input_shape[2], input_shape[3], -1)
        action_log_probs = action_log_probs.view(input_shape[0], input_shape[2], input_shape[3], -1)

        # TODO: Need to select the specific action for the relevant agents here, based on their position in the grid.

        return actions, action_log_probs, rnn_states


    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
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

        # TODO: Implement this!
        raise NotImplementedError()


class QmasCritic(R_Critic):
    ''' This class is currently the same as the regular MAPPO critic.. '''
    pass