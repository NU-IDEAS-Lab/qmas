import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space
# from onpolicy.models.diffusion.diffusion import GaussianDiffusion
# from onpolicy.models.diffusion.diffusion import ValueDiffusion
# from onpolicy.models.diffusion.temporal import TemporalUnet, ValueFunction

from onpolicy.models.diffusion.sampling.policies import GuidedPolicy
from onpolicy.models.diffusion.sampling.guides import ValueGuide
from onpolicy.models.diffusion.sampling.functions import n_step_guided_p_sample


# from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from onpolicy.models.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.classifier import OptimalityClassifier
from cleandiffuser.nn_classifier import HalfDiT1d
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
        
        share_obs_dim = get_shape_from_obs_space(cent_obs_space, flatten_dicts=False)[0] # state space for all agents
        action_dim = get_shape_from_act_space(act_space) * args.num_agents # action space for all agents
        transition_dim = share_obs_dim + action_dim

        # TODO: Need to get this null_value from the environment metadata.
        self.null_value = -1.0

        self.prediction_horizon = args.diffusion_horizon

        fix_mask = torch.zeros((self.prediction_horizon, transition_dim))
        # fix_mask[0, :obs_dim] = 1.
        # Weight actions more heavily in the loss.
        loss_weight = torch.ones((self.prediction_horizon, transition_dim))
        loss_weight[:, :action_dim] = 10.0
        
        # Create Diffuser model.
        # diffuser_base = TemporalUnet(
        #     horizon=self.prediction_horizon,
        #     transition_dim=transition_dim,
        #     cond_dim=0, #TODO: What is the correct value?
        #     dim=32,
        #     dim_mults=(8, 4, 2, 1),
        # ).to(device)
        # diffuser_base = DiT1d(
        #     x_dim=transition_dim,
        #     x_seq_len=self.prediction_horizon,
        #     emb_dim=transition_dim,
        #     d_model=256,
        #     n_heads=8,
        #     depth=4,
        #     timestep_emb_type="untrainable_fourier",
        #     timestep_emb_params={"scale": 0.02},
        # )
        diffuser_base = JannerUNet1d(
            transition_dim, model_dim=32, emb_dim=transition_dim, dim_mult=(1, 2, 4, 8),
            timestep_emb_type="positional", attention=False, kernel_size=5
        )
        # self.diffuser = GaussianDiffusion(
        #     diffuser_base, horizon = self.prediction_horizon, observation_dim = share_obs_dim, 
        #     action_dim = action_dim , n_timesteps=args.diffusion_steps, loss_type='l2', 
        #     clip_denoised=False, predict_epsilon=False,
        #     action_weight=10, loss_discount=1.0, loss_weights=None
        # ).to(device)
        
        # Create Guide model.
        # guide_base = ValueFunction(
        #     horizon = self.prediction_horizon,
        #     transition_dim = transition_dim,
        #     cond_dim=0, #TODO: What is the correct value?
        #     dim=32,
        #     dim_mults=(1, 2, 4, 8),
        #     # dim_mults=(8, 4, 2, 1),
        #     out_dim=1
        # ).to(device)
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
        # self.guide = ValueDiffusion(
        #     guide_base, horizon = self.prediction_horizon, observation_dim = share_obs_dim, 
        #     action_dim = action_dim , n_timesteps=args.diffusion_steps, loss_type='value_l2', 
        #     clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
        #     loss_discount=1.0, loss_weights=None
        # ).to(device)
        self.guide = OptimalityClassifier(guide_base).to(device)

        self.diffuser = DiscreteDiffusionSDE(diffuser_base, None, fix_mask, loss_weight, classifier=self.guide).to(device)
        # self.diffuser = ContinuousDiffusionSDE(diffuser_base, None, fix_mask, loss_weight, classifier=self.guide).to(device)

        # Update the diffuser optimizers.
        self.diffuser.manual_optimizers = {}
        self.diffuser.configure_manual_optimizers()
        
        # self.diffuser_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=2e-4)
        # self.guide_optimizer = torch.optim.Adam(self.guide.parameters(), lr=2e-4)
    

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

        # self.diffuser_guide = ValueGuide(self.guide)
        # self.diffuser_policy = GuidedPolicy(self.diffuser_guide, self.diffuser, sample_fn=n_step_guided_p_sample, conditioning_fn=self._condition_sample)


    def _condition_sample(self, x, conditions, action_dim):
        ''' Applies conditions to the sample. '''

        for t, val in conditions.items():
            # TODO: Does this need to use val.clone()? The original does, so we do as well.
            x[:, t, action_dim:] = self._overlay_tensor(x[:, t, action_dim:], val.clone(), self.null_value)
        return x
    

    def _overlay_tensor(self, a, b, null_value):
        ''' Overlays one tensor on another, based on the null value. '''

        # Determine which elements are null.
        b_data = b != null_value

        # Overlay the tensors.
        a[b_data] = b[b_data]

        return a