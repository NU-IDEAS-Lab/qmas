import numpy as np
import torch
import os.path
from .predictor import Predictor
from collections import deque

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class QmasPolicy:
    ''' This class implements the QMAS policy, including communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        
        # Create the predictor / diffusion model.
        self.obs_dim = np.prod(get_shape_from_obs_space(self.obs_space, flatten_dicts=False)) # observation space for one agent
        self.action_dim = np.prod(get_shape_from_act_space(act_space)) # action space for one agent

        if args.prediction_ensemble_size > 1:
            print(f"Creating ensemble of {args.prediction_ensemble_size} predictors.")
        self.predictors = [
            Predictor(
                self.obs_dim,
                self.action_dim,
                args,
                device=self.device
            ) for _ in range(args.prediction_ensemble_size)
        ]

        self.transition_dim = self.obs_dim + self.action_dim
        self.prediction_horizon = args.prediction_history_window
        self.buffer = deque(maxlen=self.prediction_horizon)


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


    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions for the given inputs.
        """

        # Clear the trajectory buffer if needed.
        if masks.sum() == 0:
            self.buffer.clear()
        
        # Add to the trajectory buffer. Fill up the buffer if necessary.
        obs = obs.to(self.device)
        def add_to_buffer(obs):
            # Zeros are the placeholder for the action. They will be filled in by the predictor.
            action = torch.zeros((obs.shape[0], self.action_dim), device=self.device)
            transition = torch.concatenate([action, obs], 1)
            self.buffer.append(transition)
        while len(self.buffer) < self.prediction_horizon:
            add_to_buffer(obs)
        add_to_buffer(obs)
        
        # Create the trajectory tensor from the buffer.
        trajectory = torch.stack(list(self.buffer), dim=0)  # Shape: (T, B, D)
        trajectory = trajectory.permute(1, 0, 2)  # Shape: (B, T, D)

        # Create a visibility mask.
        visibility_mask = torch.ones_like(trajectory)
        # Mark the final action as invisible, since it is not known yet.
        # This is the only part of the trajectory that will be predicted.
        visibility_mask[-1, :self.action_dim] = 0

        # Get the prediction from the ensemble of predictors.
        prediction, uncertainty = self.get_prediction(trajectory, visibility_mask, prediction_prev=None)

        # Pull the action from the prediction.
        actions = prediction[:, -1, :self.action_dim]

        # Update the trajectory buffer with the new action.
        transition = torch.concatenate([actions, obs], 1)
        self.buffer[-1] = transition  # Replace the last element.

        # Set up the returns using dummy values to match what was done in R_MAPPOPolicy.
        values = torch.zeros((self.args.n_rollout_threads, 1, 1), device=self.device)
        action_log_probs = torch.zeros((self.args.n_rollout_threads * self.args.num_agents, 1), device=self.device)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        raise NotImplementedError("The act method is not implemented for QmasPolicy.")
        # TODO: Just return the final action from the obs. This function is only called during render/eval, so the obs
        # should already contain the predicted action.


    def save(self, directory, episode):
        ''' Save the policy. '''

        super().save(directory, episode)

        for i, predictor in enumerate(self.predictors):
            torch.save(predictor.state_dict(), os.path.join(directory, f"predictor{i}.pt"))


    def restore(self, directory):
        ''' Restore the policy. '''

        super().restore(directory)

        for i, predictor in enumerate(self.predictors):
            predictor_state_dict = torch.load(os.path.join(directory, f"predictor{i}.pt"), map_location=self.device)

            # This is hacky - reset the fix_mask here.
            if 'diffuser.fix_mask' in predictor_state_dict:
                predictor_state_dict['diffuser.fix_mask'] = predictor.diffuser.fix_mask

            predictor.load_state_dict(predictor_state_dict)