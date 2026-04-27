import torch
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActorOneModule, QmasCritic


class QmasPolicy(R_MAPPOPolicy):
    """One-module QMAS policy.

    ``QmasActorOneModule`` unifies the actor and the diffusion predictor in a
    single ``nn.Module``.  A single ``actor_optimizer`` covers all parameters —
    both the policy head / prediction encoder and the diffusion backbone.

    The intermediate loss is the diffusion model's score-matching objective
    computed by ``self.actor.predictor.diffuser.loss(trajectory)``.  The
    algorithm adds this to the PPO actor loss before the single backward pass.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = QmasActorOneModule(
            args, self.obs_space, self.share_obs_space, self.act_space, self.device
        )
        self.critic = QmasCritic(args, self.share_obs_space, self.device)

        # Single optimizer covers the actor *and* the prediction encoder/decoder.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay,
        )

        # Expose the embedded predictors for the runner's UQ / observation-replacement path.
        # All predictors live on the actor's device (gradient flow requirement).
        self.predictors = list(self.actor.predictors)

    # ---------------------------------------------------------------------- #
    # Action computation                                                       #
    # ---------------------------------------------------------------------- #

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False, global_obs=None):
        """Compute actions and value predictions.

        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic,
            global_obs=global_obs,
        )

        n_agents = self.args.num_agents
        if rnn_states_critic.shape[0] == cent_obs.shape[0] * n_agents:
            critic_rnn = rnn_states_critic[::n_agents]
            critic_masks = masks[::n_agents]
            values, updated_rnn = self.critic(cent_obs, critic_rnn, critic_masks)
            rnn_states_critic = updated_rnn.repeat_interleave(n_agents, dim=0)
        else:
            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions, handling per-thread mismatch."""
        n_agents = self.args.num_agents
        if rnn_states_critic.shape[0] == cent_obs.shape[0] * n_agents:
            values, _ = self.critic(cent_obs, rnn_states_critic[::n_agents], masks[::n_agents])
        else:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def act(self, obs, rnn_states_actor, masks, available_actions=None,
            deterministic=False, global_obs=None):
        """Compute actions (no value prediction)."""
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic,
            global_obs=global_obs,
        )
        return actions, rnn_states_actor

    def restore(self, directory):
        """Restore actor and critic from a saved model directory.

        The ``fix_mask`` parameter inside each embedded predictor's diffuser is
        saved with the batch shape it had at the last training step, which is
        incompatible with the default shape initialised at construction time.
        We reset every ``fix_mask`` entry in the state dict to the currently
        initialised shape before calling ``load_state_dict``.
        """
        import os
        actor_state_dict = torch.load(
            os.path.join(directory, 'actor.pt'), map_location=self.device
        )
        # Reset any batch-sized fix_mask back to the predictor's init shape.
        for i, predictor in enumerate(self.actor.predictors):
            key = f"predictors.{i}.diffuser.fix_mask"
            if key in actor_state_dict:
                actor_state_dict[key] = predictor.diffuser.fix_mask
        self.actor.load_state_dict(actor_state_dict)

        if not self.args.use_render:
            critic_state_dict = torch.load(
                os.path.join(directory, 'critic.pt'), map_location=self.device
            )
            self.critic.load_state_dict(critic_state_dict)

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                         action, masks, available_actions=None, active_masks=None,
                         global_obs=None):
        """Compute values, action log-probs, and entropy.

        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, global_obs, rnn_states_actor, action, masks,
            available_actions, active_masks,
        )

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None,
                       has_sample_dim=False, return_member_preds=False):
        """Get a prediction from the ensemble of embedded predictors.

        Mirrors ``variant_two_modules.new.policy.QmasPolicy.get_prediction``.
        All predictors share the actor's device, so no cross-device movement is
        needed.  Uncertainty is the variance across ensemble predictions (zero
        when ``prediction_ensemble_size == 1``).

        Args:
            trajectory: Tensor of shape (T, D) or (1, T, D) when has_sample_dim=True.
            visibility_mask: Optional tensor, same shape as trajectory.
            prediction_prev: Optional previous prediction for autoregression.
            has_sample_dim: Whether trajectory already has a leading sample dim.

        Returns:
            prediction: Mean prediction across ensemble.
            uncertainty: Variance across ensemble (zeros for single predictor).
        """
        dim_ensemble = 0

        predictions = []
        for i, predictor in enumerate(self.predictors):
            member_prev = None if prediction_prev is None else prediction_prev[i].to(predictor.device)
            pred = predictor.get_prediction(
                trajectory.detach().to(predictor.device),
                visibility_mask.detach().to(predictor.device),
                member_prev,
                has_sample_dim=has_sample_dim,
            )
            predictions.append(pred.unsqueeze(0).to(self.device))
        predictions = torch.cat(predictions, dim=dim_ensemble)
        prediction = predictions.mean(dim=dim_ensemble)

        if predictions.shape[dim_ensemble] > 1:
            uncertainty = predictions.var(dim=dim_ensemble)
        else:
            uncertainty = torch.zeros_like(prediction)

        if return_member_preds:
            return prediction, uncertainty, predictions.detach()
        return prediction, uncertainty
