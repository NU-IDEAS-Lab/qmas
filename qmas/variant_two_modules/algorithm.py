import torch
import numpy as np

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.util import get_grad_norm, get_shape_from_obs_space, get_shape_from_act_space
from onpolicy.models.diffusion.diffusion import GaussianDiffusion
from onpolicy.models.diffusion.diffusion import ValueDiffusion
from onpolicy.models.diffusion.temporal import TemporalUnet, ValueFunction


class QmasAlgorithm(R_MAPPO):
    ''' This is the highest level class for the QMAS algorithm. It wraps the MAPPO algorithm and adds the diffusion model.
        This class may be wrapped by other variants (so-called 2- or 3-module) to implement communication. '''

    def __init__(self,
                 args,
                 policy,
                 env,
                 device=torch.device("cpu")):

        super().__init__(args, policy, env, device)

        self.prediction_horizon = policy.prediction_horizon


    def diffusion_update(self, diffusion_model, optimizer, loss_args, update_model):
        """
        Update diffuser network.
        :param sample: (Tuple) contains data batch with which to update networks.

        :return value_loss: (torch.Tensor) diffusion loss value.
        :return model_grad_norm: (torch.Tensor) gradient norm from model update.
        """
        if update_model:
            gradient_accumulate_every = 2
            
            # Zero the gradients.
            optimizer.zero_grad()
            
            # Accumulate gradients.
            for _ in range(gradient_accumulate_every):
                loss, info = diffusion_model.loss(*loss_args)
                loss = loss / gradient_accumulate_every

                loss.backward()
            
            # Compute gradient norm.
            model_grad_norm = get_grad_norm(diffusion_model.model.parameters())

            # Take an optimization step.
            optimizer.step()
            
            return loss.item(), model_grad_norm, info
        else:
            return None


    def train(self, buffer, update_actor=True, update_critic=True, last_step=-1):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.
        :param update_critic: (bool) whether to update critic network.
        :param last_step: (int) last step of the episode. Defaults to -1, which will use all steps in the buffer.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = super().train(buffer, update_actor, update_critic, last_step)

        num_updates = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.sample_trajectories(self.num_mini_batch, self.prediction_horizon)

            for sample in data_generator:
                self.train_sample_diffuser(sample.to(self.device), train_info, update_actor, update_critic)
                num_updates += 1

        # Average the losses.
        train_info['diffuser_loss'] /= num_updates
        train_info['guide_loss'] /= num_updates
 
        return train_info


    def train_initialize_info(self, train_info):
        super().train_initialize_info(train_info)
        train_info['diffuser_loss'] = 0
        train_info['guide_loss'] = 0


    def train_sample_diffuser(self, sample, train_info, update_actor=True, update_critic=True):
        ''' Performs update for a single sample. '''
        
        # Process shared observations.
        share_obs_batch = sample["share_obs"][:, :, :, 0] # Get the shared observation from only one agent, since they should all be the same...

        # Permute, flatten, and then permute back to get rid of the thread dimension.
        share_obs_batch = share_obs_batch.permute(1, 0, *range(2, share_obs_batch.ndim))
        share_obs_batch = share_obs_batch.flatten(start_dim=1, end_dim=2)
        share_obs_batch = share_obs_batch.permute(1, 0, *range(2, share_obs_batch.ndim))
        share_obs_batch = share_obs_batch.reshape(*share_obs_batch.shape[:2], -1)

        # Process actions.
        actions_batch = sample["actions"]
        actions_batch = actions_batch.permute(1, 0, *range(2, actions_batch.ndim))
        actions_batch = actions_batch.flatten(start_dim=1, end_dim=2)
        actions_batch = actions_batch.permute(1, 0, *range(2, actions_batch.ndim))
        actions_batch = actions_batch.reshape(*actions_batch.shape[:2], -1)

        # Process rewards.
        rewards_batch = sample["rewards"]
        rewards_batch = rewards_batch.permute(1, 0, *range(2, rewards_batch.ndim))
        rewards_batch = rewards_batch.flatten(start_dim=1, end_dim=2)
        rewards_batch = rewards_batch.permute(1, 0, *range(2, rewards_batch.ndim))
        rewards_batch = rewards_batch.reshape(*rewards_batch.shape[:2], -1)
        rewards_batch = rewards_batch.sum(axis=-1) # Sum rewards for all agents.

        # Calculate trajectory returns.
        discounts = torch.ones((rewards_batch.shape[0], rewards_batch.shape[1]), dtype=torch.float32, device=self.device) * 0.997 # TODO: This constant is from Janner et al. (2022).
        discounts = torch.pow(discounts, torch.arange(1, rewards_batch.shape[1] + 1, dtype=torch.float32, device=self.device))
        returns_batch = torch.sum(rewards_batch * discounts, dim=1).reshape((-1, 1))
        
        # TODO: Temporary placeholder for conditions. Currently empty.
        cond = {}

        # Build trajectories.
        trajectories = torch.cat([share_obs_batch, actions_batch], dim=-1)

        # Update diffuser model.
        diffuser_loss, diffuser_grad_norm, diffuser_info = self.diffusion_update(
            self.policy.diffuser,
            self.policy.diffuser_optimizer,
            (trajectories, cond),
            update_model=True
        )

        # Update guide model.
        guide_loss, guide_grad_norm, guide_info = self.diffusion_update(
            self.policy.guide,
            self.policy.guide_optimizer,
            (trajectories, cond, returns_batch),
            update_model=True
        )

        train_info['diffuser_loss'] += diffuser_loss
        train_info['guide_loss'] += guide_loss



    def prep_training(self):
        super().prep_training()
        self.policy.diffuser.train()
        self.policy.guide.train()
    

    def prep_rollout(self):
        super().prep_rollout()
        self.policy.diffuser.eval()
        self.policy.guide.eval()