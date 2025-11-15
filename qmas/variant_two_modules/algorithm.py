import torch
import numpy as np
import threading

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.util import get_grad_norm, get_shape_from_obs_space, get_shape_from_act_space


class QmasAlgorithm(R_MAPPO):
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
        self.predictors = policy.predictors
        self.num_predictors = len(self.predictors)
        self.prediction_horizon = self.predictors[0].prediction_horizon  # Assume all predictors have same horizon

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

        def policy_train_thread():
            nonlocal policy_info
            policy_info = super(QmasAlgorithm, self).train(buffer, update_actor, update_critic, last_step)

        def predictor_train_thread():
            nonlocal predictor_info
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
                            # Use a local dict to accumulate losses for this predictor
                            local_info = {'diffuser_loss': 0, 'guide_loss': 0}
                            if len(thread_splits[i]) == 0:
                                raise ValueError("Thread split is empty. Check prediction_ensemble_size and n_rollout_threads.")
                            self.train_sample_diffuser(sample, local_info, predictor, thread_indices=thread_splits[i])
                            pred_losses[i] = local_info['diffuser_loss']
                            pred_guides[i] = local_info['guide_loss']
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

        threads = []
        # Start policy training thread
        t_policy = threading.Thread(target=policy_train_thread)
        threads.append(t_policy)
        t_policy.start()

        # Start predictor training thread if enabled
        if update_predictor:
            t_predictor = threading.Thread(target=predictor_train_thread)
            threads.append(t_predictor)
            t_predictor.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Merge results
        train_info.update(policy_info)
        if update_predictor:
            train_info['diffuser_loss'] = predictor_info['diffuser_loss']
            train_info['guide_loss'] = predictor_info['guide_loss']

        return train_info

    def train_sample_diffuser(self, sample, train_info, predictor, thread_indices=None):
        ''' Performs update for a single sample for a given predictor. '''
        
        # Permute, flatten, and then permute back to get rid of the thread dimension.

        # Process observations.
        if thread_indices is None:
            obs_batch = sample["share_obs"]
        else:
            obs_batch = sample["share_obs"][:, :, thread_indices]
        obs_batch = obs_batch.permute(1, 0, *range(2, obs_batch.ndim))
        obs_batch = obs_batch.flatten(start_dim=1, end_dim=2)
        obs_batch = obs_batch.permute(1, 0, *range(2, obs_batch.ndim))
        obs_batch = obs_batch.reshape(*obs_batch.shape[:2], -1)

        # Process rewards.
        if thread_indices is None:
            rewards_batch = sample["rewards"]
        else:
            rewards_batch = sample["rewards"][:, :, thread_indices]
        rewards_batch = rewards_batch.permute(1, 0, *range(2, rewards_batch.ndim))
        rewards_batch = rewards_batch.flatten(start_dim=1, end_dim=2)
        rewards_batch = rewards_batch.permute(1, 0, *range(2, rewards_batch.ndim))
        rewards_batch = rewards_batch.reshape(*rewards_batch.shape[:2], -1)

        # Sum rewards across all agents.
        rewards_batch = torch.sum(rewards_batch, dim=-1)

        # Condition using visibility mask.
        if thread_indices is None:
            visibility_mask_batch = sample["state_visibility_mask"]
        else:
            visibility_mask_batch = sample["state_visibility_mask"][:, :, thread_indices]
        visibility_mask_batch = visibility_mask_batch.permute(1, 0, *range(2, visibility_mask_batch.ndim))
        visibility_mask_batch = visibility_mask_batch.flatten(start_dim=1, end_dim=2)
        visibility_mask_batch = visibility_mask_batch.permute(1, 0, *range(2, visibility_mask_batch.ndim))
        visibility_mask_batch = visibility_mask_batch.reshape(*visibility_mask_batch.shape[:2], -1)
        fix_mask_batch = visibility_mask_batch.float()
        
        # Calculate trajectory returns.
        discounts = torch.ones((rewards_batch.shape[0], rewards_batch.shape[1]), dtype=torch.float32) * 0.997 # TODO: This constant is from Janner et al. (2022).
        discounts = torch.pow(discounts, torch.arange(1, rewards_batch.shape[1] + 1, dtype=torch.float32))
        returns_batch = torch.sum(rewards_batch * discounts, dim=1).reshape((-1, 1))

        # Build trajectories.
        # trajectories = torch.cat([actions_batch, obs_batch], dim=-1)
        trajectories = obs_batch

        # Get the visibility mask for the current agent.

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

        # Update guide model.
        # guide_loss = predictor.diffuser.update_classifier(
        #     x0=trajectories,
        #     condition_cg=returns_batch
        # )['classifier_loss']

        train_info['diffuser_loss'] += diffuser_loss
        # train_info['guide_loss'] += guide_loss

    def prep_training(self):
        super().prep_training()
        for predictor in self.predictors:
            predictor.train()    

    def prep_rollout(self):
        super().prep_rollout()
        for predictor in self.predictors:
            predictor.eval()
