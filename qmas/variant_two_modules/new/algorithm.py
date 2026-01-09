import torch
import threading
import sys
import einops

from onpolicy.models.utils.util import check
from onpolicy.utils.util import get_grad_norm
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as Algorithm


class QmasAlgorithm(Algorithm):
    ''' This is the highest level class for the QMAS algorithm. It wraps the MAPPO algorithm and adds the diffusion model.
        This class may be wrapped by other variants (so-called 2- or 3-module) to implement communication. 
        Modified to support an ensemble of predictors, each trained on a different set of threads.
    '''

    def __init__(self,
                 args,
                 policy,
                 env,
                 device=torch.device("cpu")):

        super().__init__(args, policy, env, device)

        # Assume policy.predictors is a list of predictor modules (ensemble)
        self.use_threads = args.threaded_training
        self.predictors = policy.predictors
        self.num_predictors = len(self.predictors)
        self.prediction_horizon = self.predictors[0].prediction_horizon  # Assume all predictors have same horizon
        self.predictor_2d_conv = args.diffusion_model_type == "unet2d"

        print(f"Initialized QmasAlgorithm with {self.num_predictors} predictors. Threaded training: {self.use_threads}")

    def train(self, buffer, update_actor=True, update_critic=True, last_step=-1, episode=None, episodes=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.
        :param update_critic: (bool) whether to update critic network.
        :param last_step: (int) last step of the episode. Defaults to -1, which will use all steps in the buffer.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        """
        Perform training in two serial phases:
        1. Policy training (actor-critic)
        2. Diffusion model training (ensemble)
        """
        
        train_info = {}
        self.train_initialize_info(train_info)

        # Thread exception capture: collect exceptions raised in any thread
        thread_exceptions = []
        thread_exceptions_lock = threading.Lock()

        # Determine what to update.
        update_predictor = not self.args.prediction_disable
        if episode is not None and episodes is not None:
            frac = episode / episodes
            if frac < self.args.episode_fraction_start_prediction:
                update_predictor = False
            if frac > self.args.episode_fraction_stop_policy:
                update_actor = False
                update_critic = False

        # Results containers for threads
        policy_info = {}
        predictor_info = {'diffuser_loss': 0, 'guide_loss': 0}

        def train_policy():
            nonlocal policy_info
            try:
                policy_info = super(QmasAlgorithm, self).train(buffer, update_actor, update_critic, last_step)
            except Exception:
                with thread_exceptions_lock:
                    thread_exceptions.append(sys.exc_info())

        def train_predictor():
            nonlocal predictor_info
            try:
                num_diffusion_updates = [0 for _ in range(self.num_predictors)]
                assert self.args.n_rollout_threads % self.args.prediction_ensemble_size == 0, "n_rollout_threads must be divisible by prediction_ensemble_size."
                split_size = self.args.n_rollout_threads // self.args.prediction_ensemble_size
                thread_indices = torch.arange(self.args.n_rollout_threads)
                thread_splits = torch.split(thread_indices, split_size)

                for e in range(self.args.diffusion_epoch):
                    data_generator = buffer.sample_trajectories(self.num_mini_batch, self.prediction_horizon)
                    for sample in data_generator:
                        # For each sample, run each predictor's update in its own thread
                        pred_threads = []
                        pred_losses = [None] * self.num_predictors
                        pred_guides = [None] * self.num_predictors

                        def make_pred_thread(i, predictor):
                            def run():
                                try:
                                    # Use a local dict to accumulate losses for this predictor
                                    local_info = {'diffuser_loss': 0, 'guide_loss': 0}
                                    if len(thread_splits[i]) == 0:
                                        raise ValueError("Thread split is empty. Check prediction_ensemble_size and n_rollout_threads.")
                                    self.train_sample_diffuser(sample, local_info, predictor, thread_indices=thread_splits[i])
                                    pred_losses[i] = local_info['diffuser_loss']
                                    pred_guides[i] = local_info['guide_loss']
                                except Exception:
                                    with thread_exceptions_lock:
                                        thread_exceptions.append(sys.exc_info())
                            return run

                        for i, predictor in enumerate(self.predictors):
                            t = threading.Thread(target=make_pred_thread(i, predictor))
                            pred_threads.append(t)
                            t.start()

                        for t in pred_threads:
                            t.join()

                        for i in range(self.num_predictors):
                            if pred_losses[i] is not None:
                                predictor_info['diffuser_loss'] += pred_losses[i]
                                predictor_info['guide_loss'] += pred_guides[i]
                                num_diffusion_updates[i] += 1

                total_updates = sum(num_diffusion_updates)
                if total_updates > 0:
                    predictor_info['diffuser_loss'] /= total_updates
                    predictor_info['guide_loss'] /= total_updates
            except Exception:
                with thread_exceptions_lock:
                    thread_exceptions.append(sys.exc_info())

        threads = []

        if self.use_threads:
            # Start policy training
            t_policy = threading.Thread(target=train_policy)
            threads.append(t_policy)
            t_policy.start()
        else:
            train_policy()
        
        # Start predictor training if enabled
        if update_predictor:
            if self.use_threads:
                t_predictor = threading.Thread(target=train_predictor)
                threads.append(t_predictor)
                t_predictor.start()
            else:
                train_predictor()

        # Wait for all threads to finish
        if self.use_threads:
            for t in threads:
                t.join()

        # If any thread captured an exception, re-raise the first one in the main thread
        if thread_exceptions:
            exc_type, exc_value, exc_tb = thread_exceptions[0]
            raise exc_value.with_traceback(exc_tb)

        # Merge results
        train_info.update(policy_info)
        if update_predictor:
            train_info['diffuser_loss'] = predictor_info['diffuser_loss']
            train_info['guide_loss'] = predictor_info['guide_loss']

        return train_info

    def train_sample_diffuser(self, sample, train_info, predictor, thread_indices=None):
        ''' Performs update for a single sample for a given predictor. '''
        
        # If thread indices given, use only data from those threads.
        if thread_indices is None:
            obs_batch = sample["share_obs"]
            rewards_batch = sample["rewards"]
            fix_mask_batch = sample["state_visibility_mask"]
        else:
            obs_batch = sample["share_obs"][:, :, thread_indices]
            rewards_batch = sample["rewards"][:, :, thread_indices]
            fix_mask_batch = sample["state_visibility_mask"][:, :, thread_indices]

        if self.predictor_2d_conv:
            # Set up batches for 2D convolution.
            obs_batch = einops.rearrange(obs_batch, 'b h t c ... -> (b t) (h c) ...')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t c ... -> (b t) (h c) ...')
        else:
            # Set up batches for 1D convolution.
            obs_batch = einops.rearrange(obs_batch, 'b h t ... -> (b t) h (...)')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t ... -> (b t) h (...)')

        # Process rewards. Sum rewards across all agents.
        rewards_batch = einops.rearrange(rewards_batch, 'b h t ... -> (b t) h ...')
        rewards_batch = torch.sum(rewards_batch, dim=-1)

        # Calculate trajectory returns.
        discounts = torch.ones((rewards_batch.shape[0], rewards_batch.shape[1]), dtype=torch.float32) * 0.997 # TODO: This constant is from Janner et al. (2022).
        discounts = torch.pow(discounts, torch.arange(1, rewards_batch.shape[1] + 1, dtype=torch.float32))
        discounts = discounts.unsqueeze(-1)
        returns_batch = torch.sum(rewards_batch * discounts, dim=1) #.reshape((-1, 1))

        # Build trajectories.
        # trajectories = torch.cat([actions_batch, obs_batch], dim=-1)
        trajectories = obs_batch

        # Transfer tensors to the device.
        trajectories = trajectories.to(predictor.device)
        returns_batch = returns_batch.to(predictor.device)
        fix_mask_batch = fix_mask_batch.to(predictor.device)

        # Update the fix_mask. This determines which parts of the trajectory are fixed and which are predicted.
        # This applies to both update_diffusion and update_classifier.
        predictor.diffuser.fix_mask = torch.nn.Parameter(fix_mask_batch, requires_grad=False)

        # Update diffuser model.
        diffuser_loss = predictor.diffuser.update_diffusion(
            x0=trajectories,
        )['diffusion_loss']
        train_info['diffuser_loss'] += diffuser_loss

        # Update guide model.
        if predictor.diffuser.classifier is not None:
            guide_loss = predictor.diffuser.update_classifier(
                x0=trajectories,
                condition_cg=returns_batch
            )['classifier_loss']
            train_info['guide_loss'] += guide_loss


    def ppo_update(self, sample, update_actor=True, update_critic=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.
        :update_critic: (bool) whether to update critic network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, global_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

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
                                                                              active_masks_batch,
                                                                              global_obs=global_obs_batch)
        
        # Reshape and repeat values for each agent.
        # The value function predictions are made once over the entire share_obs.
        # However, we need to compare them with per-agent returns.
        values = values.reshape((values.shape[0], 1, 1)).repeat(1, self.policy.args.num_agents, 1).reshape((-1, 1))

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

        # Calculate overall loss for actor.
        if update_actor:
            self.policy.actor_optimizer.zero_grad()
            actor_loss = policy_loss - dist_entropy * self.entropy_coef
        else:
            actor_loss = 0.0

        # Calculate overall loss for critic.
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch, update_value_normalizer=update_critic)
        if update_critic:
            self.policy.critic_optimizer.zero_grad()
            critic_loss = value_loss * self.value_loss_coef
        else:
            critic_loss = 0.0
        
        # Combine loss and backpropagate gradients.
        loss = actor_loss + critic_loss
        if update_actor or update_critic:
            loss.backward()

        # Clip gradient norms.
        if self._use_max_grad_norm:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        if self._use_max_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        # Step the optimizers.
        if update_actor:
            self.policy.actor_optimizer.step()
        if update_critic:
            self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights


    def prep_training(self):
        super().prep_training()
        for predictor in self.predictors:
            predictor.train()    

    def prep_rollout(self):
        super().prep_rollout()
        for predictor in self.predictors:
            predictor.eval()
