import torch

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.util import get_grad_norm, get_shape_from_obs_space, get_shape_from_act_space
from onpolicy.models.utils.diffuser import GaussianDiffusion
from onpolicy.models.utils.guide import ValueDiffusion


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
        self.observation_dim = get_shape_from_obs_space(env.observation_space[0], flatten_dicts=False)
        self.action_dim = get_shape_from_act_space(env.action_space[0])
        
        self.diffuser = GaussianDiffusion(horizon = 32, observation_dim = self.observation_dim, 
                                          action_dim = self.action_dim , n_timesteps=20, model = None, loss_type='l2', 
                                          clip_denoised=False, predict_epsilon=False,
                                          action_weight=10, loss_discount=1.0, loss_weights=None)
        
        self.guide = ValueDiffusion(horizon = 32, observation_dim = self.observation_dim, 
                                    action_dim = self.action_dim , n_timesteps=20, model = None, loss_type='value_l2', 
                                    clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
                                    loss_discount=1.0, loss_weights=None) 


    def diffusion_update(self, diffusion_model, sample, update_model):
        """
        Update diffuser network.
        :param sample: (Tuple) contains data batch with which to update networks.

        :return value_loss: (torch.Tensor) diffusion loss value.
        :return model_grad_norm: (torch.Tensor) gradient norm from model update.
        """
        if update_model:
            gradient_accumulate_every = 2
            diffusion_optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=2e-4)
            
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
            
            cond = [()]

            # Combine observations and actions for the diffuser input
            # Assuming obs_batch and actions_batch are properly shaped
            obs_batch = torch.from_numpy(obs_batch)
            actions_batch = torch.from_numpy(actions_batch)
            trajectories = torch.cat([obs_batch, actions_batch], dim=-1)
        
            for i in range(gradient_accumulate_every):
                batch_size = trajectories.shape[0]
                t = torch.randint(0, diffusion_model.n_timesteps, (batch_size,), device=trajectories.device).long()
                
                loss, info = diffusion_model.loss(trajectories, share_obs_batch, t)
                loss = loss / gradient_accumulate_every
                loss.backward()
            
            model_grad_norm = get_grad_norm(diffusion_model.model.parameters())

            # Optimizer step
            diffusion_optimizer.step()
            diffusion_optimizer.zero_grad()

            return loss.item(), model_grad_norm, info
        else:
            return None

    def train_initialize_info(self, train_info):
        super().train_initialize_info(train_info)
        train_info['diffuser_loss'] = 0
        train_info['guide_loss'] = 0


    def train_sample(self, sample, train_info, update_actor=True, update_critic=True):
        ''' Performs update for a single sample. '''
        
        super().train_sample(sample, train_info, update_actor, update_critic)

        diffuser_loss, model_grad_norm, info = self.diffusion_update(self.diffuser, sample, update_model=True)
        guide_loss, model_grad_norm, info = self.diffusion_update(self.guide, sample, update_model=True)

        train_info['diffuser_loss'] += diffuser_loss.item()
        train_info['guide_loss'] += guide_loss.item()



    def prep_training(self):
        super().prep_training()
        self.diffuser.train()
        self.guide.train()
    

    def prep_rollout(self):
        super().prep_rollout()
        self.diffuser.eval()
        self.guide.eval()