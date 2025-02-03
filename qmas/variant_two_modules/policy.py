import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from onpolicy.models.diffusion.diffusion import GaussianDiffusion
from onpolicy.models.diffusion.diffusion import ValueDiffusion
from onpolicy.models.diffusion.temporal import TemporalUnet, ValueFunction


class QmasPolicy(R_MAPPOPolicy):
    ''' This class implements the QMAS policy, including communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # We use the QMAS actor, but the default MAPPO critic.
        self.actor = QmasActor(args, self.obs_space, self.act_space, self.device)
        self.critic = QmasCritic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        share_obs_dim = get_shape_from_obs_space(cent_obs_space, flatten_dicts=False)[0] # state space for all agents
        action_dim = get_shape_from_act_space(act_space) * args.num_agents # action space for all agents
        transition_dim = share_obs_dim + action_dim
        
        # Create Diffuser model.
        self.prediction_horizon = 8
        diffuser_base = TemporalUnet(
            horizon=self.prediction_horizon,
            transition_dim=transition_dim,
            cond_dim=0, #TODO: What is the correct value?
            dim=32,
            dim_mults=(8, 4, 2, 1),
        ).to(device)
        self.diffuser = GaussianDiffusion(
            diffuser_base, horizon = self.prediction_horizon, observation_dim = share_obs_dim, 
            action_dim = action_dim , n_timesteps=4, loss_type='l2', 
            clip_denoised=False, predict_epsilon=False,
            action_weight=10, loss_discount=1.0, loss_weights=None
        ).to(device)
        
        # Create Guide model.
        guide_base = ValueFunction(
            horizon = self.prediction_horizon,
            transition_dim = transition_dim,
            cond_dim=0, #TODO: What is the correct value?
            dim=32,
            dim_mults=(8, 4, 2, 1),
            out_dim=1
        ).to(device)
        self.guide = ValueDiffusion(
            guide_base, horizon = self.prediction_horizon, observation_dim = share_obs_dim, 
            action_dim = action_dim , n_timesteps=4, loss_type='value_l2', 
            clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
            loss_discount=1.0, loss_weights=None
        ).to(device)
        
        self.diffuser_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=2e-4)
        self.guide_optimizer = torch.optim.Adam(self.guide.parameters(), lr=2e-4)
    

    def save(self, directory, episode):
        ''' Save the policy. '''

        super().save(directory, episode)

        torch.save(self.diffuser.state_dict(), os.path.join(directory, "diffuser.pt"))
        torch.save(self.guide.state_dict(), os.path.join(directory, "guide.pt"))
    

    def restore(self, directory):
        ''' Restore the policy. '''

        super().restore(directory)

        diffuser_state_dict = torch.load(os.path.join(directory, 'diffuser.pt'))
        self.diffuser.load_state_dict(diffuser_state_dict)

        guide_state_dict = torch.load(os.path.join(directory, 'guide.pt'))
        self.guide.load_state_dict(guide_state_dict)