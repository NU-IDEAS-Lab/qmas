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
        self.prediction_horizon = args.prediction_history_window
        self.predictor_2d_conv = args.diffusion_model_type == "unet2d"

        self._pred_streams = [
            torch.cuda.Stream(device=p.device) if p.device.type == "cuda" else None
            for p in self.predictors
        ]

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
        update_predictor = not self.args.prediction_disable and self.predictors is not None and len(self.predictors) > 0
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

        # Results containers for threads
        policy_info = {}
        predictor_info = {'diffuser_loss': 0, 'guide_loss': 0, 'uncertainty_loss': 0}

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

                pred_streams = self._pred_streams

                # One-shot seed of normalization stats from a large sample, before any
                # diffuser updates run. Without this, running mean/var crawl up from the
                # zero-mean / unit-var prior over many minibatches and the diffuser
                # chases a moving normalization target.
                seed_targets = [i for i, p in enumerate(self.predictors) if not bool(p.normalizer_seeded.item())]
                if seed_targets:
                    seed_sample = next(buffer.sample_trajectories(1, self.prediction_horizon))
                    for i in seed_targets:
                        predictor = self.predictors[i]
                        if len(thread_splits[i]) == 0:
                            continue
                        seed_trajs, _, _ = self._build_training_trajectories(
                            seed_sample, predictor, thread_indices=thread_splits[i]
                        )
                        predictor.seed_normalization_stats(seed_trajs)

                for e in range(self.args.diffusion_epoch):
                    data_generator = buffer.sample_trajectories(self.num_mini_batch, self.prediction_horizon)
                    for sample in data_generator:
                        # For each sample, run each predictor's update in its own thread and CUDA stream.
                        pred_threads = []
                        pred_losses = [None] * self.num_predictors
                        pred_guides = [None] * self.num_predictors
                        pred_uncerts = [None] * self.num_predictors

                        def make_pred_thread(i, predictor, stream):
                            def run():
                                try:
                                    local_info = {'diffuser_loss': 0, 'guide_loss': 0, 'uncertainty_loss': 0}
                                    if len(thread_splits[i]) == 0:
                                        raise ValueError("Thread split is empty. Check prediction_ensemble_size and n_rollout_threads.")
                                    if stream is not None:
                                        with torch.cuda.device(predictor.device):
                                            # Make this stream wait for any prior work queued on
                                            # the default stream (e.g. data already on GPU).
                                            stream.wait_stream(torch.cuda.current_stream(stream.device))
                                            with torch.cuda.stream(stream):
                                                self.train_sample_diffuser(sample, local_info, predictor, thread_indices=thread_splits[i])
                                            # Default stream needs to wait on this one before
                                            # downstream readers (e.g. the policy thread or next epoch).
                                            torch.cuda.current_stream(stream.device).wait_stream(stream)
                                    else:
                                        self.train_sample_diffuser(sample, local_info, predictor, thread_indices=thread_splits[i])
                                    pred_losses[i] = local_info['diffuser_loss']
                                    pred_guides[i] = local_info['guide_loss']
                                    pred_uncerts[i] = local_info.get('uncertainty_loss', 0)
                                except Exception:
                                    with thread_exceptions_lock:
                                        thread_exceptions.append(sys.exc_info())
                            return run

                        for i, predictor in enumerate(self.predictors):
                            t = threading.Thread(target=make_pred_thread(i, predictor, pred_streams[i]))
                            pred_threads.append(t)
                            t.start()

                        for t in pred_threads:
                            t.join()

                        for i in range(self.num_predictors):
                            if pred_losses[i] is not None:
                                predictor_info['diffuser_loss'] += pred_losses[i]
                                predictor_info['guide_loss'] += pred_guides[i]
                                predictor_info['uncertainty_loss'] += pred_uncerts[i]
                                num_diffusion_updates[i] += 1

                total_updates = sum(num_diffusion_updates)
                if total_updates > 0:
                    predictor_info['diffuser_loss'] /= total_updates
                    predictor_info['guide_loss'] /= total_updates
                    predictor_info['uncertainty_loss'] /= total_updates
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
            train_info['uncertainty_loss'] = predictor_info['uncertainty_loss']

        return train_info

    def _build_training_trajectories(self, sample, predictor, thread_indices=None):
        ''' Build (trajectories, fix_mask_batch, returns_batch) the way the diffuser
            consumes them. Extracted from train_sample_diffuser so other callers
            (e.g. one-shot normalization seeding) can reuse the same construction. '''

        if "state_visibility_mask" in sample:
            key_obs = "share_obs"
            key_visibility = "state_visibility_mask"
            individual_obs = False
        else:
            key_obs = "obs_full"
            key_visibility = "visibility_mask"
            individual_obs = True

        # Raise a clear error if obs_full holds non-tensor (e.g. PyG graph) objects. In this case, the environment must provide global_visibility_mask.
        if isinstance(sample[key_obs], list):
            raise ValueError(
                "obs_full contains non-tensor (PyG graph) observations and no "
                "'state_visibility_mask' was found in the sample.  Set "
                "observe_method_global='adjacency' in the environment so that it "
                "provides state_visibility_mask and the diffuser can train on the "
                "float adjacency global state instead of the graph observations."
            )

        # If thread indices given, use only data from those threads.
        if thread_indices is None:
            obs_batch = sample[key_obs]
            rewards_batch = sample["rewards"]
            fix_mask_batch = sample[key_visibility]
            actions_batch = sample["actions"]
        else:
            obs_batch = sample[key_obs][:, :, thread_indices]
            rewards_batch = sample["rewards"][:, :, thread_indices]
            fix_mask_batch = sample[key_visibility][:, :, thread_indices]
            actions_batch = sample["actions"][:, :, thread_indices]

        # If using individual observations, we need to treat each agent as a separate batch item.
        if individual_obs:
            obs_batch = einops.rearrange(obs_batch, 'b h t n ... -> (b n) h t ...')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t n ... -> (b n) h t ...')
            actions_batch = einops.rearrange(actions_batch, 'b h t n ... -> (b n) h t ...')
        elif self.args.prediction_history_include_actions:
            # For global-state observations (individual_obs=False), the diffuser was initialised with
            # transition_dim = per-agent action_dim + obs_dim.  At inference time, each
            # (thread × agent) pair contributes its own trajectory, so we need to replicate
            # the global obs once per agent and pair it with each agent's action before
            # applying the standard per-agent rearrange.
            n_agents = actions_batch.shape[3]  # [b, h, t, n_agents, action_dim]
            obs_batch = obs_batch.unsqueeze(3).expand(-1, -1, -1, n_agents, -1)   # [b, h, t, n, obs_dim]
            fix_mask_batch = fix_mask_batch.unsqueeze(3).expand(-1, -1, -1, n_agents, -1)
            obs_batch = einops.rearrange(obs_batch, 'b h t n ... -> (b n) h t ...')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t n ... -> (b n) h t ...')
            actions_batch = einops.rearrange(actions_batch, 'b h t n ... -> (b n) h t ...')

        if self.predictor_2d_conv:
            # Set up batches for 2D convolution.
            obs_batch = einops.rearrange(obs_batch, 'b h t c ... -> (b t) (h c) ...')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t c ... -> (b t) (h c) ...')
            actions_batch = einops.rearrange(actions_batch, 'b h t c ... -> (b t) (h c) ...')
        else:
            # Set up batches for 1D convolution.
            obs_batch = einops.rearrange(obs_batch, 'b h t ... -> (b t) h (...)')
            fix_mask_batch = einops.rearrange(fix_mask_batch, 'b h t ... -> (b t) h (...)')
            actions_batch = einops.rearrange(actions_batch, 'b h t ... -> (b t) h (...)')

        # Process rewards. Sum rewards across all agents.
        rewards_batch = einops.rearrange(rewards_batch, 'b h t ... -> (b t) h ...')
        rewards_batch = torch.sum(rewards_batch, dim=-1)

        # Calculate trajectory returns.
        discounts = torch.ones((rewards_batch.shape[0], rewards_batch.shape[1]), dtype=torch.float32) * 0.997 # TODO: This constant is from Janner et al. (2022).
        discounts = torch.pow(discounts, torch.arange(1, rewards_batch.shape[1] + 1, dtype=torch.float32))
        discounts = discounts.unsqueeze(-1)
        returns_batch = torch.sum(rewards_batch * discounts, dim=1) #.reshape((-1, 1))

        # Build trajectories.
        if self.args.prediction_history_include_actions:
            trajectories = torch.cat([actions_batch, obs_batch], dim=-1)
            # Pad fix_mask with ones for action dimensions (actions are always observed).
            action_mask = torch.ones_like(actions_batch)
            fix_mask_batch = torch.cat([action_mask, fix_mask_batch], dim=-1)
        else:
            trajectories = obs_batch

        # Transfer tensors to the device.
        trajectories = trajectories.float().to(predictor.device)
        returns_batch = returns_batch.float().to(predictor.device)
        fix_mask_batch = fix_mask_batch.float().to(predictor.device)

        return trajectories, fix_mask_batch, returns_batch

    def train_sample_diffuser(self, sample, train_info, predictor, thread_indices=None):
        ''' Performs update for a single sample for a given predictor. '''

        trajectories, fix_mask_batch, returns_batch = self._build_training_trajectories(
            sample, predictor, thread_indices=thread_indices
        )

        # Update running stats from training trajectories and train in normalized space.
        predictor.update_normalization_stats(trajectories)
        trajectories_normalized = predictor.normalize_trajectory(trajectories)

        # Update the fix_mask. This determines which parts of the trajectory are fixed and which are predicted.
        # This applies to both update_diffusion and update_classifier.
        predictor.diffuser.fix_mask = fix_mask_batch

        # Update diffuser model.
        diffuser_loss = predictor.diffuser.update_diffusion(
            x0=trajectories_normalized,
        )['diffusion_loss']
        train_info['diffuser_loss'] += diffuser_loss

        # Calculate uncertainty estimation loss.
        if self.args.prediction_uq_method == "estimation":
            # Get prediction.
            predictions = predictor.get_prediction(
                trajectories,
                visibility_mask=fix_mask_batch,
                has_sample_dim=True,
            ).detach()
            predictions_error = predictions - trajectories

            predictions_masked = predictions * fix_mask_batch
            predicted_lb, predicted_ub = predictor.get_uncertainty_bounds(predictions_masked)
            uncertainty_loss = predictor.uncertainty_bounds_estimator.loss(
                predictions_error.flatten(start_dim=1),
                predicted_lb,
                predicted_ub
            )
            # Backpropagate and update uncertainty estimator parameters.
            predictor.uncertainty_optimizer.zero_grad()
            uncertainty_loss.backward()
            predictor.uncertainty_optimizer.step()

            train_info['uncertainty_loss'] += uncertainty_loss.item()

        # Update guide model.
        if predictor.diffuser.classifier is not None:
            guide_loss = predictor.diffuser.update_classifier(
                x0=trajectories_normalized,
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
        # In the recurrent case share_obs is already indexed per-agent, so values
        # already matches adv_targ in size — skip the repeat.
        if values.shape[0] != adv_targ.shape[0]:
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
