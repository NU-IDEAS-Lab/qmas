import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.classifier import OptimalityClassifier
from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d


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
        
        obs_dim = get_shape_from_obs_space(self.obs_space, flatten_dicts=False)[0] # state space for all agents
        action_dim = get_shape_from_act_space(act_space) * args.num_agents # action space for all agents
        transition_dim = obs_dim + action_dim

        self.prediction_horizon = args.diffusion_horizon

        fix_mask = torch.zeros((self.prediction_horizon, transition_dim))

        # Weight actions more heavily in the loss.
        loss_weight = torch.ones((self.prediction_horizon, transition_dim))
        loss_weight[:, :action_dim] = 1.0
        
        # Create Diffuser model.
        print(f"QmasPolicy: Using diffusion model type {args.diffusion_model_type} with prediction horizon {self.prediction_horizon} and transition dimension {transition_dim}.")
        if args.diffusion_model_type == "jannerunet":
            diffuser_base = JannerUNet1d(
                transition_dim, model_dim=32, emb_dim=32, dim_mult=(1, 2, 4, 8),
                timestep_emb_type="positional",
                attention=False, kernel_size=5
            )
            guide_base = HalfJannerUNet1d(
                horizon=self.prediction_horizon,
                in_dim=transition_dim,
                out_dim=1,
                model_dim=32,
                emb_dim=32,
                dim_mult=(1, 2, 4, 8),
                timestep_emb_type="positional"
            )
        elif args.diffusion_model_type == "dit1d":
            diffuser_base = DiT1d(
                x_dim=transition_dim,
                x_seq_len=self.prediction_horizon,
                emb_dim=128,
                d_model=256,
                n_heads=8,
                depth=4,
                timestep_emb_type="untrainable_fourier",
                timestep_emb_params={"scale": 0.02},
            )
            guide_base = HalfDiT1d(
                x_dim=transition_dim,
                out_dim=1,
                x_seq_len=self.prediction_horizon,
                emb_dim=128,
                d_model=256,
                n_heads=8,
                depth=4,
                timestep_emb_type="untrainable_fourier",
                timestep_emb_params={"scale": 0.02},
            )
        else:
            raise ValueError(f"Unknown diffusion model type: {args.diffusion_model_type}")
        
        # Create the guide and diffuser.
        self.guide = OptimalityClassifier(guide_base).to(device)
        self.diffuser = DiscreteDiffusionSDE(
            diffuser_base,
            None,
            fix_mask,
            loss_weight,
            classifier=self.guide,
            predict_noise=False
        ).to(device)

        # Update the diffuser optimizers.
        self.diffuser.manual_optimizers = {}
        self.diffuser.configure_manual_optimizers()


    def save(self, directory, episode):
        ''' Save the policy. '''

        super().save(directory, episode)

        torch.save(self.diffuser.state_dict(), os.path.join(directory, "diffuser.pt"))
        torch.save(self.guide.state_dict(), os.path.join(directory, "guide.pt"))


    def restore(self, directory):
        ''' Restore the policy. '''

        super().restore(directory)

        diffuser_state_dict = torch.load(os.path.join(directory, 'diffuser.pt'), map_location=self.device)

        # This is hacky - reset the fix_mask here.
        if 'fix_mask' in diffuser_state_dict:
            diffuser_state_dict['fix_mask'] = self.diffuser.fix_mask

        self.diffuser.load_state_dict(diffuser_state_dict)

        guide_state_dict = torch.load(os.path.join(directory, 'guide.pt'), map_location=self.device)
        self.guide.load_state_dict(guide_state_dict)