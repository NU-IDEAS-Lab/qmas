import torch
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic

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