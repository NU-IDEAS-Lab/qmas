import torch

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
        share_obs_dim = get_shape_from_obs_space(env.share_observation_space[0], flatten_dicts=False)[0]
        action_dim = get_shape_from_act_space(env.action_space[0])
        transition_dim = share_obs_dim + action_dim
        
        # Create Diffuser model.
        horizon = 32 # TODO: Need to feed in 32-length trajectories during training.
        diffuser_base = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim,
            cond_dim=0, #TODO: What is the correct value?
            dim=32,
            # dim_mults=(2, 1)
            # dim_mults=(8, 4, 2, 1)
            dim_mults=(1, 2, 4, 8)
        )
        self.diffuser = GaussianDiffusion(diffuser_base, horizon = horizon, observation_dim = share_obs_dim, 
                                          action_dim = action_dim , n_timesteps=4, loss_type='l2', 
                                          clip_denoised=False, predict_epsilon=False,
                                          action_weight=10, loss_discount=1.0, loss_weights=None)
        
        # Create Guide model.
        guide_base = ValueFunction(
            horizon = horizon,
            transition_dim = transition_dim,
            cond_dim=0, #TODO: What is the correct value?
            dim=32,
            dim_mults=(8, 4, 2, 1),
            out_dim=1
        )
        self.guide = ValueDiffusion(guide_base, horizon = horizon, observation_dim = share_obs_dim, 
                                    action_dim = action_dim , n_timesteps=4, loss_type='value_l2', 
                                    clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
                                    loss_discount=1.0, loss_weights=None)
        
        self.diffuser_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=2e-4)
        self.guide_optimizer = torch.optim.Adam(self.guide.parameters(), lr=2e-4)


    def diffusion_update(self, diffusion_model, optimizer, sample, update_model):
        """
        Update diffuser network.
        :param sample: (Tuple) contains data batch with which to update networks.

        :return value_loss: (torch.Tensor) diffusion loss value.
        :return model_grad_norm: (torch.Tensor) gradient norm from model update.
        """
        if update_model:
            gradient_accumulate_every = 2
            
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
            
            # TODO: Temporary placeholder for conditions. Currently empty.
            cond = {}

            # Build trajectory.
            # Combine observations and actions for the diffuser input
            # Assuming obs_batch and actions_batch are properly shaped
            share_obs_batch = torch.from_numpy(share_obs_batch).unsqueeze(1)
            actions_batch = torch.from_numpy(actions_batch).unsqueeze(1)
            trajectories = torch.cat([share_obs_batch, actions_batch], dim=-1)

            # TODO: Major problem: The above batch size does not match the diffusion model's horizon size.
            # I believe that these must match.
            # TODO: Temporarily just repeat along dimension 1 to match the horizon size.
            trajectories = trajectories.repeat(1, 32, 1)
            print(f"Trajectories shape: {trajectories.shape}")

            # Zero the gradients.
            optimizer.zero_grad()
            
            # Accumulate gradients.
            for _ in range(gradient_accumulate_every):
                loss, info = diffusion_model.loss(trajectories, cond)
                loss = loss / gradient_accumulate_every

                loss.backward()
            
            # Compute gradient norm.
            model_grad_norm = get_grad_norm(diffusion_model.model.parameters())

            # Take an optimization step.
            optimizer.step()
            
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

        diffuser_loss, model_grad_norm, info = self.diffusion_update(self.diffuser, self.diffuser_optimizer, sample, update_model=True)

        # TODO: NEED TO PASS IN THE TARGET (RETURNS?) FOR THE VALUE FUNCTION.
        raise NotImplementedError("Need to pass in the target for the value function.")
        guide_loss, model_grad_norm, info = self.diffusion_update(self.guide, self.guide_optimizer, sample, update_model=True)

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