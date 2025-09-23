import torch
from qmas.variant_two_modules.algorithm import QmasAlgorithm

class QmasAlgorithm(QmasAlgorithm):
    """
    Trainer class for QMAS to update policies.
    """

    def train_sample_diffuser(self, sample, train_info, predictor, thread_indices=None):
        '''
        Performs update for a single sample for a given predictor.
        Same as the parent but only takes the communication actions.
        '''
        
        # Permute, flatten, and then permute back to get rid of the thread dimension.

        # Process observations.
        if thread_indices is None:
            obs_batch = sample["obs_full"]
        else:
            obs_batch = sample["obs_full"][:, :, thread_indices]
        obs_batch = obs_batch.permute(1, 0, *range(2, obs_batch.ndim))
        obs_batch = obs_batch.flatten(start_dim=1, end_dim=2)
        obs_batch = obs_batch.permute(1, 0, *range(2, obs_batch.ndim))
        obs_batch = obs_batch.reshape(*obs_batch.shape[:3], -1)

        # Process actions.
        if thread_indices is None:
            actions_batch = sample["actions"]
        else:
            actions_batch = sample["actions"][:, :, thread_indices]
        
        actions_batch = actions_batch[:, :, :, :, self.policy.action_comm_indices]

        actions_batch = actions_batch.permute(1, 0, *range(2, actions_batch.ndim))
        actions_batch = actions_batch.flatten(start_dim=1, end_dim=2)
        actions_batch = actions_batch.permute(1, 0, *range(2, actions_batch.ndim))
        actions_batch = actions_batch.reshape(*actions_batch.shape[:3], -1)

        # Process rewards.
        if thread_indices is None:
            rewards_batch = sample["rewards"]
        else:
            rewards_batch = sample["rewards"][:, :, thread_indices]
        rewards_batch = rewards_batch.permute(1, 0, *range(2, rewards_batch.ndim))
        rewards_batch = rewards_batch.flatten(start_dim=1, end_dim=2)
        rewards_batch = rewards_batch.permute(1, 0, *range(2, rewards_batch.ndim))
        rewards_batch = rewards_batch.reshape(*rewards_batch.shape[:2], -1)

        # Condition using visibility mask.
        if thread_indices is None:
            visibility_mask_batch = sample["visibility_mask"]
        else:
            visibility_mask_batch = sample["visibility_mask"][:, :, thread_indices]
        visibility_mask_batch = visibility_mask_batch.permute(1, 0, *range(2, visibility_mask_batch.ndim))
        visibility_mask_batch = visibility_mask_batch.flatten(start_dim=1, end_dim=2)
        visibility_mask_batch = visibility_mask_batch.permute(1, 0, *range(2, visibility_mask_batch.ndim))
        visibility_mask_batch = visibility_mask_batch.reshape(*visibility_mask_batch.shape[:3], -1)
        
        action_visibility = torch.ones_like(actions_batch)
        fix_mask_batch = torch.cat([action_visibility, visibility_mask_batch.float()], dim=-1)
        
        # Perform optimization step for all agents.
        for i in range(actions_batch.shape[2]):
            agent_obs_batch = obs_batch[:, :, i]
            agent_actions_batch = actions_batch[:, :, i]
            agent_rewards_batch = rewards_batch[:, :, i]

            # Calculate trajectory returns.
            discounts = torch.ones((agent_rewards_batch.shape[0], agent_rewards_batch.shape[1]), dtype=torch.float32) * 0.997 # TODO: This constant is from Janner et al. (2022).
            discounts = torch.pow(discounts, torch.arange(1, rewards_batch.shape[1] + 1, dtype=torch.float32))
            agent_returns_batch = torch.sum(agent_rewards_batch * discounts, dim=1).reshape((-1, 1))

            # Build trajectories.
            trajectories = torch.cat([agent_actions_batch, agent_obs_batch], dim=-1)

            # Get the visibility mask for the current agent.
            agent_fix_mask = fix_mask_batch[:, :, i]

            # Transfer tensors to the device.
            trajectories = trajectories.to(self.device)
            agent_returns_batch = agent_returns_batch.to(self.device)
            agent_fix_mask = agent_fix_mask.to(self.device)

            # Update the fix_mask. This determines which parts of the trajectory are fixed and which are predicted.
            # This applies to both update_diffusion and update_classifier.
            predictor.diffuser.fix_mask = torch.nn.Parameter(agent_fix_mask, requires_grad=False)

            # Update diffuser model.
            diffuser_loss = predictor.diffuser.update_diffusion(
                x0=trajectories,
            )['diffusion_loss']

            # Update guide model.
            guide_loss = predictor.diffuser.update_classifier(
                x0=trajectories,
                condition_cg=agent_returns_batch
            )['classifier_loss']

            train_info['diffuser_loss'] += diffuser_loss
            train_info['guide_loss'] += guide_loss