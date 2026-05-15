import torch
import torch.nn as nn
import einops
from itertools import cycle

from onpolicy.models.utils.util import check
from onpolicy.utils.util import get_grad_norm
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO


class QmasAlgorithmOneModule(R_MAPPO):
    """Training algorithm for the one-module QMAS policy.

    The actor and the diffusion predictor are a single ``nn.Module``; both are
    optimised by a single ``actor_optimizer``.

    For each PPO mini-batch update ``ppo_update`` performs **one backward
    pass** that combines:

    1. The standard MAPPO PPO loss (clipped surrogate + value loss).
    2. The diffusion model's **score-matching loss** computed by
       ``predictor.diffuser.loss(trajectory_batch)``.  A trajectory mini-batch
       is drawn from a pre-sampled pool at the start of each training epoch so
       that one diffusion loss is fused with every PPO loss before
       ``backward()`` is called.

    After ``actor_optimizer.step()`` the EMA model of the diffuser is updated
    via ``predictor.diffuser.ema_update()`` so that rollout inference (which
    uses ``use_ema=True``) stays current.

    The weight of the diffusion loss is controlled by
    ``args.prediction_loss_coef`` (default 1.0).
    """

    def __init__(self, args, policy, env, device=torch.device("cpu")):
        super().__init__(args, policy, env, device)

        self.prediction_loss_coef = args.prediction_loss_coef
        self.predictor_2d_conv = args.diffusion_model_type == "unet2d"

        # Will be populated at the start of each train() call.
        self._trajectory_iter = None
        self._update_predictor_this_epoch = False

        print(
            f"Initialized QmasAlgorithmOneModule. "
            f"prediction_loss_coef={self.prediction_loss_coef}, "
            f"prediction_disable={args.prediction_disable}"
        )

    # ---------------------------------------------------------------------- #
    # Override train() to set up trajectory iterator and gating flags         #
    # ---------------------------------------------------------------------- #

    def train(self, buffer, update_actor=True, update_critic=True,
              last_step=-1, episode=None, episodes=None):
        """Run PPO + diffusion score-matching training.

        A pool of trajectory mini-batches is sampled once at the start of this
        call and made available to ``ppo_update`` via a cycling iterator.
        This keeps the diffusion loss in the same backward pass as each PPO
        update without requiring trajectory data to be stuffed into the PPO
        sample tuple.
        """
        update_predictor = not self.args.prediction_disable
        if episode is not None and episodes is not None:
            frac = episode / episodes
            if frac < self.args.episode_fraction_start_prediction:
                update_predictor = False
            if frac < self.args.episode_fraction_start_policy:
                update_actor = False
                update_critic = False
            if frac > self.args.episode_fraction_stop_policy:
                update_actor = False
                update_critic = False

        self._update_predictor_this_epoch = update_predictor

        if update_predictor:
            predictor = self.policy.actor.predictors[0]
            horizon = predictor.prediction_horizon
            traj_batches = list(buffer.sample_trajectories(self.num_mini_batch, horizon))
            self._trajectory_iter = cycle(traj_batches) if traj_batches else None
        else:
            self._trajectory_iter = None

        return super().train(
            buffer,
            update_actor=update_actor,
            update_critic=update_critic,
            last_step=last_step,
            episode=episode,
            episodes=episodes,
        )

    # ---------------------------------------------------------------------- #
    # Override train_initialize_info to add diffusion_loss key                #
    # ---------------------------------------------------------------------- #

    def train_initialize_info(self, train_info):
        super().train_initialize_info(train_info)
        train_info['diffuser_loss'] = torch.zeros(1, device=self.device)

    # ---------------------------------------------------------------------- #
    # Override train_sample to accumulate diffusion_loss                       #
    # ---------------------------------------------------------------------- #

    def train_sample(self, sample, train_info, update_actor=True, update_critic=True):
        value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, \
            imp_weights, diffusion_loss = self.ppo_update(sample, update_actor, update_critic)

        train_info['value_loss'] += value_loss
        train_info['policy_loss'] += policy_loss
        train_info['dist_entropy'] += dist_entropy
        train_info['actor_grad_norm'] += actor_grad_norm
        train_info['critic_grad_norm'] += critic_grad_norm
        train_info['ratio'] += imp_weights.mean()
        train_info['diffuser_loss'] += torch.tensor(diffusion_loss, device=self.device)

    # ---------------------------------------------------------------------- #
    # Core PPO + diffusion update (single backward pass)                       #
    # ---------------------------------------------------------------------- #

    def ppo_update(self, sample, update_actor=True, update_critic=True):
        """Update actor (including diffusion predictor) and critic.

        The diffusion predictor is a submodule of ``self.policy.actor``.  Its
        parameters are registered in ``self.policy.actor_optimizer``.  We
        obtain a differentiable score-matching loss from
        ``predictor.diffuser.loss(trajectory_batch)`` and add it to the PPO
        actor loss **before** calling ``backward()``.  This gives a single
        gradient pass that simultaneously updates:

        - The policy head and prediction encoder (via PPO gradients).
        - The diffusion backbone (via score-matching gradients).

        After the optimizer step, ``predictor.diffuser.ema_update()`` keeps
        the EMA model current for rollout inference.

        Returns:
            value_loss, critic_grad_norm, policy_loss, dist_entropy,
            actor_grad_norm, imp_weights, diffusion_loss_value (float)
        """
        (share_obs_batch, obs_batch, global_obs_batch,
         rnn_states_batch, rnn_states_critic_batch,
         actions_batch, value_preds_batch, return_batch,
         masks_batch, active_masks_batch,
         old_action_log_probs_batch, adv_targ,
         available_actions_batch) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # PPO forward pass (no prediction tensor returned).
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch,
            rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch,
            available_actions_batch, active_masks_batch,
            global_obs=global_obs_batch,
        )

        # Align value shape with per-agent advantages.
        if values.shape[0] != adv_targ.shape[0]:
            values = (values.reshape(values.shape[0], 1, 1)
                      .repeat(1, self.policy.args.num_agents, 1)
                      .reshape(-1, 1))

        # ------------------------------------------------------------------ #
        # PPO actor loss                                                       #
        # ------------------------------------------------------------------ #
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # ------------------------------------------------------------------ #
        # Diffusion score-matching loss                                        #
        #                                                                      #
        # A trajectory mini-batch is drawn from the cycling iterator set up   #
        # in train().  The batch is preprocessed the same way as in           #
        # variant_two_modules.new.algorithm.train_sample_diffuser, then       #
        # passed to predictor.diffuser.loss() which returns a differentiable  #
        # scalar score-matching loss.  This is added to the actor loss so     #
        # gradient flows through the diffusion backbone in the same backward. #
        # ------------------------------------------------------------------ #
        update_predictor = self._update_predictor_this_epoch
        diffusion_loss_value = 0.0

        if update_predictor and self._trajectory_iter is not None:
            traj_sample = next(self._trajectory_iter)
            diffusion_loss = self._compute_diffusion_loss(traj_sample)
            diffusion_loss_value = diffusion_loss.item()
        else:
            diffusion_loss = torch.zeros(1, **self.tpdv)

        # ------------------------------------------------------------------ #
        # Critic (value) loss                                                  #
        # ------------------------------------------------------------------ #
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch,
            update_value_normalizer=update_critic,
        )

        # ------------------------------------------------------------------ #
        # Single combined backward pass                                        #
        # ------------------------------------------------------------------ #
        if update_actor:
            self.policy.actor_optimizer.zero_grad()
            actor_loss = (policy_action_loss
                          - dist_entropy * self.entropy_coef
                          + self.prediction_loss_coef * diffusion_loss)
        else:
            actor_loss = 0.0

        if update_critic:
            self.policy.critic_optimizer.zero_grad()
            critic_loss = value_loss * self.value_loss_coef
        else:
            critic_loss = 0.0

        total_loss = actor_loss + critic_loss
        if update_actor or update_critic:
            total_loss.backward()

        # Gradient clipping.
        if self._use_max_grad_norm:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

        if self._use_max_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        if update_actor:
            self.policy.actor_optimizer.step()

        if update_critic:
            self.policy.critic_optimizer.step()

        # Keep the diffusion EMA model current after each optimizer step.
        if update_predictor and update_actor:
            for predictor in self.policy.actor.predictors:
                predictor.diffuser.ema_update()

        return (value_loss, critic_grad_norm, policy_action_loss,
                dist_entropy, actor_grad_norm, imp_weights, diffusion_loss_value)

    # ---------------------------------------------------------------------- #
    # Diffusion loss computation helper                                        #
    # ---------------------------------------------------------------------- #

    def _compute_diffusion_loss(self, traj_sample) -> torch.Tensor:
        """Compute the differentiable diffusion score-matching loss.

        Mirrors the batch preparation logic of
        ``variant_two_modules.new.algorithm.train_sample_diffuser`` but uses
        all rollout threads (no ensemble splitting) and returns the raw loss
        tensor rather than calling an internal optimizer step.

        Args:
            traj_sample: TensorDict from ``buffer.sample_trajectories``.
                Shape per key: ``(batch, horizon, n_rollout_threads, ...)``.

        Returns:
            Differentiable scalar loss tensor on the actor's device.
        """
        predictors = self.policy.actor.predictors

        if "state_visibility_mask" in traj_sample:
            key_obs = "share_obs"
            key_visibility = "state_visibility_mask"
            individual_obs = False
        else:
            key_obs = "obs_full"
            key_visibility = "visibility_mask"
            individual_obs = True

        if isinstance(traj_sample[key_obs], list):
            raise ValueError(
                "obs_full contains non-tensor (PyG graph) observations and no "
                "'state_visibility_mask' was found in the sample.  Set "
                "observe_method_global='adjacency' in the environment so that it "
                "provides state_visibility_mask."
            )

        obs_batch = traj_sample[key_obs]         # (b, h, t, ...)
        fix_mask_batch = traj_sample[key_visibility]  # (b, h, t, ...)

        # Per-agent individual observations need agent dim merged into batch.
        if individual_obs:
            obs_batch = einops.rearrange(obs_batch, 'b h t n ... -> (b n) h t ...')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t n ... -> (b n) h t ...')

        if self.predictor_2d_conv:
            obs_batch = einops.rearrange(obs_batch, 'b h t c ... -> (b t) (h c) ...')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t c ... -> (b t) (h c) ...')
        else:
            obs_batch = einops.rearrange(obs_batch, 'b h t ... -> (b t) h (...)')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t ... -> (b t) h (...)')

        obs_batch = obs_batch.float().to(self.device)
        fix_mask_batch = fix_mask_batch.float().to(self.device)

        total_loss = torch.zeros(1, **self.tpdv)
        for predictor in predictors:
            # Update running normalization stats (no_grad, purely bookkeeping).
            predictor.update_normalization_stats(obs_batch)
            norm_obs = predictor.normalize_trajectory(obs_batch)

            # Assign per-batch fix_mask so diffuser.loss() masks the unseen region.
            predictor.diffuser.fix_mask = nn.Parameter(fix_mask_batch, requires_grad=False)

            # Differentiable score-matching loss.
            total_loss = total_loss + predictor.diffuser.loss(x0=norm_obs)

        return total_loss / len(predictors)

    # ---------------------------------------------------------------------- #
    # prep_training / prep_rollout – actor handles predictor via submodule    #
    # ---------------------------------------------------------------------- #

    def prep_training(self):
        super().prep_training()
        # predictor is a submodule of actor; actor.train() covers it already.

    def prep_rollout(self):
        super().prep_rollout()
        # predictor is a submodule of actor; actor.eval() covers it already.

