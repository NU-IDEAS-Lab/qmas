import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

# import config
from onpolicy.diffuser.config import diffuser_config
from onpolicy.diffuser.config import guide_config

# import trainer
from onpolicy.diffuser.utils.training import Trainer
    
class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                #  diffuser_config,
                #  guide_config,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.diffuser_config = diffuser_config
        self.guide_config = guide_config

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        # Diffusion
        ## diffuser
        self.diffuser_dataloader = diffuser_config.loader(
                                        env=diffuser_config.dataset,
                                        horizon=diffuser_config.horizon,
                                        normalizer=diffuser_config.normalizer,
                                        preprocess_fns=diffuser_config.preprocess_fns,
                                        use_padding=diffuser_config.use_padding,
                                        max_path_length=diffuser_config.max_path_length,
                                    )
        self.diffuser_observation_dim = self.diffuser_dataloader.observation_dim
        self.diffuser_action_dim = self.diffuser_dataloader.action_dim
        self.diffuser_model = diffuser_config.base_model(
                                        horizon=diffuser_config.horizon,
                                        transition_dim=self.diffuser_observation_dim + self.diffuser_action_dim,
                                        cond_dim=self.diffuser_observation_dim,
                                        dim_mults=diffuser_config.dim_mults,
                                        attention=diffuser_config.attention
                                    )
        self.diffuser_diffusion_model = diffuser_config.diffusion_model(
                                        horizon=diffuser_config.horizon,
                                        observation_dim=self.diffuser_observation_dim,
                                        action_dim=self.diffuser_action_dim,
                                        transition_dim=self.diffuser_observation_dim + self.diffuser_action_dim,
                                        model = self.diffuser_model,
                                        n_timesteps=diffuser_config.n_diffusion_steps,
                                        loss_type=diffuser_config.loss_type,
                                        clip_denoised=diffuser_config.clip_denoised,
                                        predict_epsilon=diffuser_config.predict_epsilon,
                                        action_weight=diffuser_config.action_weight,
                                        loss_weights=diffuser_config.loss_weights,
                                        loss_discount=diffuser_config.loss_discount
                                    )
        self.diffuser_trainer = Trainer(
                                        diffusion_model = self.diffuser_diffusion_model,
                                        dataset = self.diffuser_dataloader
                                        train_batch_size=diffuser_config.batch_size,
                                        train_lr=diffuser_config.learning_rate,
                                        gradient_accumulate_every=diffuser_config.gradient_accumulate_every,
                                        ema_decay=diffuser_config.ema_decay,
                                        sample_freq=diffuser_config.sample_freq,
                                        save_freq=diffuser_config.save_freq,
                                        log_freq=diffuser_config.lof_freq,
                                        label_freq=int(diffuser_config.n_train_steps // diffuser_config.n_saves),
                                        save_parallel=diffuser_config.save_parallel,
                                        bucket=diffuser_config.bucket,
                                        n_reference=diffuser_config.n_reference,
                                    )
                                            
        
        ## guide
        self.guide_dataloader = guide_config.loader(
                                        env=guide_config.dataset,
                                        horizon=guide_config.horizon,
                                        normalizer=guide_config.normalizer,
                                        preprocess_fns=guide_config.preprocess_fns,
                                        use_padding=guide_config.use_padding,
                                        max_path_length=guide_config.max_path_length,
                                    )
        self.guide_observation_dim = self.guide_dataloader.observation_dim
        self.guide_action_dim = self.guide_dataloader.action_dim
        self.guide_model = guide_config.base_model(
                                        horizon=guide_config.horizon,
                                        transition_dim=self.guide_observation_dim + self.guide_action_dim,
                                        cond_dim=self.guide_observation_dim,
                                        dim_mults=guide_config.dim_mults,
                                        attention=guide_config.attention
                                    )
        self.guide_diffusion_model = guide_config.diffusion_model(
                                        horizon=guide_config.horizon,
                                        observation_dim=self.guide_observation_dim,
                                        action_dim=self.guide_action_dim,
                                        transition_dim=self.guide_observation_dim + self.guide_action_dim,
                                        model = self.guide_model,
                                        n_timesteps=guide_config.n_diffusion_steps,
                                        loss_type=guide_config.loss_type,
                                        clip_denoised=guide_config.clip_denoised,
                                        predict_epsilon=guide_config.predict_epsilon,
                                        action_weight=guide_config.action_weight,
                                        loss_weights=guide_config.loss_weights,
                                        loss_discount=guide_config.loss_discount
                                    )
        self.guide_trainer = Trainer(
                                        diffusion_model = self.guide_diffusion_model,
                                        dataset = self.guide_dataloader
                                        train_batch_size=guide_config.batch_size,
                                        train_lr=guide_config.learning_rate,
                                        gradient_accumulate_every=guide_config.gradient_accumulate_every,
                                        ema_decay=guide_config.ema_decay,
                                        sample_freq=guide_config.sample_freq,
                                        save_freq=guide_config.save_freq,
                                        log_freq=guide_config.lof_freq,
                                        label_freq=int(guide_config.n_train_steps // guide_config.n_saves),
                                        save_parallel=guide_config.save_parallel,
                                        bucket=guide_config.bucket,
                                        n_reference=guide_config.n_reference,
                                    )
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        
        
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights


    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['diffuser_loss'] = 0
        train_info['guide_loss'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

            # for _ in range(n_epochs): 
                diffuser_loss = self.diffuser_trainer.train()
                guide_loss = self.guide_trainer.train()
                
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['diffuser_loss'] += diffuser_loss.item()
                train_info['guide_loss'] = += guide_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
