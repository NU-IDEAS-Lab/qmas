import torch
import torch.nn as nn
import gymnasium.spaces as spaces
import numpy as np

from onpolicy.models.utils.cnn import CNNBase
from onpolicy.models.utils.mlp import MLPBase
from onpolicy.models.utils.util import init
from onpolicy.utils.util import get_shape_from_obs_space
from ..actor_critic import R_Actor as Actor, R_Critic as Critic

class StateEncoder(nn.Module):
    ''' This class encodes the global state for communication purposes. '''

    def __init__(self, args, state_space, output_dim, use_ReLU, use_orthogonal, device=torch.device("cpu"), gain=None):
        super(StateEncoder, self).__init__()
        self.device = device
        self.args = args
        hidden_size = args.hidden_size

        obs_shape = get_shape_from_obs_space(state_space)
        self._use_cnn = len(obs_shape) == 3

        input_dim = np.prod(obs_shape)

        if self._use_cnn:
            self.cnn = CNNBase(args, obs_shape, mode="encoder")
            input_dim = hidden_size
        else:
            self.mlp = MLPBase(args, input_dim)
            input_dim = hidden_size

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        if gain is None:
            gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.fc = init_(nn.Linear(input_dim, output_dim))
        self.active_func = active_func

    def forward(self, state):
        if self._use_cnn:
            x = self.cnn(state)
        else:
            x = self.mlp(state)
        x = self.fc(x)
        x = self.active_func(x)
        return x


class QmasActor(Actor):
    ''' This class modifies the original QMAS actor to include a state encoder. '''
    
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        # Enhance the observation space to include encoded global state.
        enhanced_obs_space = spaces.Dict({
            'obs': obs_space,
            'encoded_state': spaces.Box(low=-np.inf, high=np.inf, shape=(args.state_encoder_output_dim,), dtype=np.float32)
        })
        enhanced_obs_space = spaces.flatten_space(enhanced_obs_space)
        
        super().__init__(args, enhanced_obs_space, act_space, device)

        # Create the state encoder
        self.state_encoder = StateEncoder(
            args,
            share_obs_space,
            output_dim=args.state_encoder_output_dim,
            use_ReLU=args.use_ReLU,
            use_orthogonal=args.use_orthogonal,
            device=device
        )


    def forward(self, obs, global_obs, rnn_states, masks, available_actions=None, deterministic=False):
        if global_obs == None:
            # If no global state is provided, use zeros.
            batch_size = obs.shape[0]
            encoded_state = torch.zeros((batch_size, self.args.state_encoder_output_dim), device=obs.device)
        else:
            # Encode the global state
            encoded_state = self.state_encoder(global_obs)

            # Repeat the encoded state for each agent if necessary.
            if encoded_state.shape[0] < obs.shape[0]:
                if obs.shape[0] % encoded_state.shape[0] != 0:
                    raise ValueError("Batch size of obs is not a multiple of batch size of share_obs.")
                encoded_state = encoded_state.repeat_interleave(obs.shape[0] // encoded_state.shape[0], dim=0)
        
        # Flatten the observation.
        obs = torch.flatten(obs, start_dim=1)

        # Concatenate the encoded state to the original observation
        enhanced_obs = torch.cat([obs, encoded_state], dim=-1)

        # Pass the enhanced observation to the original actor forward method
        return super().forward(enhanced_obs, rnn_states, masks, available_actions, deterministic)


    def evaluate_actions(self, obs, global_obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if global_obs == None:
            # If no global state is provided, use zeros.
            batch_size = obs.shape[0]
            encoded_state = torch.zeros((batch_size, self.args.state_encoder_output_dim), device=obs.device)
        else:
            # Encode the global state
            encoded_state = self.state_encoder(global_obs)

            # Repeat the encoded state for each agent if necessary.
            if encoded_state.shape[0] < obs.shape[0]:
                if obs.shape[0] % encoded_state.shape[0] != 0:
                    raise ValueError("Batch size of obs is not a multiple of batch size of share_obs.")
                encoded_state = encoded_state.repeat_interleave(obs.shape[0] // encoded_state.shape[0], dim=0)

        # Flatten the observation.
        obs = torch.flatten(obs, start_dim=1)

        # Concatenate the encoded state to the original observation
        enhanced_obs = torch.cat([obs, encoded_state], dim=-1)

        # Pass the enhanced observation to the original actor evaluate_actions method
        return super().evaluate_actions(enhanced_obs, rnn_states, action, masks, available_actions, active_masks)


class QmasCritic(Critic):
    ''' This class is currently the same as the regular MAPPO critic.. '''
    pass