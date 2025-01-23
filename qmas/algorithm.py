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

        self.env = env
        share_obs_dim = get_shape_from_obs_space(env.share_observation_space[0], flatten_dicts=False)[0] # state space for all agents
        action_dim = get_shape_from_act_space(env.action_space[0]) * len(env.action_space) # action space for all agents
        transition_dim = share_obs_dim + action_dim
        
        # Create Diffuser model.
        self.prediction_horizon = 8
        diffuser_base = TemporalUnet(
            horizon=self.prediction_horizon,
            transition_dim=transition_dim,
            cond_dim=0, #TODO: What is the correct value?
            dim=32,
            dim_mults=(8, 4, 2, 1)
        )
        self.diffuser = GaussianDiffusion(diffuser_base, horizon = self.prediction_horizon, observation_dim = share_obs_dim, 
                                          action_dim = action_dim , n_timesteps=4, loss_type='l2', 
                                          clip_denoised=False, predict_epsilon=False,
                                          action_weight=10, loss_discount=1.0, loss_weights=None)
        
        # Create Guide model.
        guide_base = ValueFunction(
            horizon = self.prediction_horizon,
            transition_dim = transition_dim,
            cond_dim=0, #TODO: What is the correct value?
            dim=32,
            dim_mults=(8, 4, 2, 1),
            out_dim=1
        )
        self.guide = ValueDiffusion(guide_base, horizon = self.prediction_horizon, observation_dim = share_obs_dim, 
                                    action_dim = action_dim , n_timesteps=4, loss_type='value_l2', 
                                    clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
                                    loss_discount=1.0, loss_weights=None)
        
        self.diffuser_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=2e-4)
        self.guide_optimizer = torch.optim.Adam(self.guide.parameters(), lr=2e-4)


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
                self.train_sample_diffuser(sample, train_info, update_actor, update_critic)
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
        
        # Unpack sample.
        share_obs_batch = sample["share_obs"][:, :, 0, 0] # Get the shared observation from only one agent, since they should all be the same...
        share_obs_batch = share_obs_batch.reshape(*share_obs_batch.shape[:2], -1)
        actions_batch = sample["actions"].reshape(*sample["actions"].shape[:2], -1)
        returns_batch = sample["returns"].reshape(*sample["returns"].shape[:2], -1)
        
        # TODO: Temporary placeholder for conditions. Currently empty.
        cond = {}

        # Build trajectories.
        trajectories = torch.cat([share_obs_batch, actions_batch], dim=-1)

        # Update diffuser model.
        diffuser_loss, diffuser_grad_norm, diffuser_info = self.diffusion_update(
            self.diffuser,
            self.diffuser_optimizer,
            (trajectories, cond),
            update_model=True
        )

        # Sum the returns.
        # TODO: This needs to be the return over each trajectory, not the sum of all trajectories.
        # Should end up with shape [batch_size, 1].
        # TODO: This is currently invalid.
        returns = returns_batch.sum(axis=(-1, -2))

        # Update guide model.
        guide_loss, guide_grad_norm, guide_info = self.diffusion_update(
            self.guide,
            self.guide_optimizer,
            (trajectories, cond, returns),
            update_model=True
        )

        train_info['diffuser_loss'] += diffuser_loss
        train_info['guide_loss'] += guide_loss



    def prep_training(self):
        super().prep_training()
        self.diffuser.train()
        self.guide.train()
    

    def prep_rollout(self):
        super().prep_rollout()
        self.diffuser.eval()
        self.guide.eval()