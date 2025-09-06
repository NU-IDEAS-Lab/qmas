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
        self.predictors = [Predictor(
            obs_dim,
            action_dim,
            args,
            device=self.device
        )]


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None):
        """
        Get a prediction from the ensemble of predictors.
        Args:
            trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
            visibility_mask: An optional tensor of shape (T,) indicating which timesteps are visible (1) or not (0).
            prediction_prev: An optional tensor of shape (T, D_out) representing the previous prediction to condition on.
        Returns:
            prediction: A tensor of shape (T, D_out) representing the mean prediction across the ensemble.
            uncertainty: A tensor of shape (T, D_out) representing the uncertainty (variance) across the ensemble predictions.
        """

        predictions = []
        for predictor in self.predictors:
            pred = predictor.get_prediction(trajectory, visibility_mask, prediction_prev)
            predictions.append(pred.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)  # Shape: (num_predictors, T, D_out)
        prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)

        return prediction, uncertainty
