import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic
from .predictor import Predictor

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class QmasPolicy(R_MAPPOPolicy):
    ''' This class implements the QMAS policy, including communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

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
        
        # Create the predictor / diffusion model.
        obs_dim = get_shape_from_obs_space(self.obs_space, flatten_dicts=False)[0] # observation space for one agent
        action_dim = get_shape_from_act_space(act_space) # action space for one agent
        self.predictor = Predictor(
            obs_dim,
            action_dim,
            args,
            device=self.device
        )


    def save(self, directory, episode):
        ''' Save the policy. '''

        super().save(directory, episode)

        torch.save(self.predictor.state_dict(), os.path.join(directory, "predictor.pt"))


    def restore(self, directory):
        ''' Restore the policy. '''

        super().restore(directory)

        diffuser_state_dict = torch.load(os.path.join(directory, 'predictor.pt'), map_location=self.device)

        # This is hacky - reset the fix_mask here.
        if 'diffuser.fix_mask' in diffuser_state_dict:
            diffuser_state_dict['diffuser.fix_mask'] = self.predictor.diffuser.fix_mask

        self.predictor.load_state_dict(diffuser_state_dict)