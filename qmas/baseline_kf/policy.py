import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .predictor import Predictor

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class QmasPolicy(R_MAPPOPolicy):
    ''' This class implements the QMAS policy, including communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        # Initialize the base MAPPO policy.
        super().__init__(args, obs_space, cent_obs_space, act_space, device)

        # Create the predictor / diffusion model.
        obs_dim = get_shape_from_obs_space(self.obs_space, flatten_dicts=False)[0] # observation space for one agent
        action_dim = get_shape_from_act_space(act_space) # action space for one agent
        self.predictor = Predictor(
            obs_dim,
            action_dim,
            args,
            device=self.device
        )