import numpy as np
import torch
from ..policy import QmasPolicy as Policy
from .actor_critic import QmasActor, QmasCritic
from .predictor import Predictor

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class QmasPolicy(Policy):
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

        # We use the QMAS actor, but the default MAPPO critic.
        self.actor = QmasActor(args, self.obs_space, self.share_obs_space, self.act_space, self.device)
        self.critic = QmasCritic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        # Create the predictor / diffusion model.
        share_obs_shape = get_shape_from_obs_space(self.share_obs_space, flatten_dicts=False)
        share_obs_dim = np.prod(share_obs_shape) # observation space for one agent
        action_dim = np.prod(get_shape_from_act_space(act_space)) # action space for one agent

        if args.prediction_ensemble_size > 1:
            print(f"Creating ensemble of {args.prediction_ensemble_size} predictors.")
        
        self.predictors = []
        for i in range(args.prediction_ensemble_size):
            if args.cuda and torch.cuda.is_available():
                if len(args.cuda_idx_predictor) > 1:
                    device_predictor = torch.device(f"cuda:{args.cuda_idx_predictor[i]}")
                elif len(args.cuda_idx_predictor) == 1:
                    device_predictor = torch.device(f"cuda:{args.cuda_idx_predictor[0]}")
                else:
                    device_predictor = self.device
            else:
                device_predictor = self.device

            predictor = Predictor(
                share_obs_dim,
                action_dim,
                args,
                obs_shape=share_obs_shape,
                device=device_predictor
            ).to(device_predictor)
            self.predictors.append(predictor)


    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, global_obs=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 global_obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, global_obs=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, global_obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, global_obs=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     global_obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy


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
            if prediction_prev is not None:
                prediction_prev = prediction_prev.to(predictor.device)
            pred = predictor.get_prediction(
                trajectory.clone().to(predictor.device),
                visibility_mask.clone().to(predictor.device),
                prediction_prev)
            predictions.append(pred.unsqueeze(0).to(self.device))
        predictions = torch.cat(predictions, dim=0)  # Shape: (num_predictors, T, D_out)
        prediction = predictions.mean(dim=0)

        uncertainty_lb, uncertainty_ub = predictor.get_uncertainty_bounds(trajectory.unsqueeze(0))
        uncertainty = torch.abs(uncertainty_ub - uncertainty_lb)

        return prediction, uncertainty
