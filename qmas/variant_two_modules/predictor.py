import torch

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.classifier import OptimalityClassifier
from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d, UNet2d


class Predictor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, args, device=None, obs_shape=None):
        super(Predictor, self).__init__()
        self.args = args
        self.device = device

        # transition_dim = obs_dim + action_dim
        transition_dim = obs_dim

        self.prediction_horizon = args.prediction_history_window

        # Set up initial fixed mask and loss weight.
        fix_mask = torch.zeros((self.prediction_horizon, transition_dim))
        # Weight actions more heavily in the loss.
        loss_weight = torch.ones((self.prediction_horizon, transition_dim))
        # loss_weight[:, :action_dim] = 1.0
        
        # Create Diffuser model.
        print(f"QmasPolicy: Using diffusion model type {args.diffusion_model_type} with prediction horizon {self.prediction_horizon} and transition dimension {transition_dim}.")
        if args.diffusion_model_type == "jannerunet":
            diffuser_base = JannerUNet1d(
                transition_dim, model_dim=32, emb_dim=32, dim_mult=(1, 2),
                timestep_emb_type="positional",
                attention=False, kernel_size=3
            )
            guide_base = HalfJannerUNet1d(
                horizon=self.prediction_horizon,
                in_dim=transition_dim,
                out_dim=1,
                model_dim=32,
                emb_dim=32,
                dim_mult=(1, 2),
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
        elif args.diffusion_model_type == "unet2d":
            # Update transition dim. Don't use actions.
            transition_dim = obs_dim
            loss_weight = torch.ones((self.prediction_horizon * obs_shape[0], obs_shape[1], obs_shape[2]))

            diffuser_base = UNet2d(
                n_channels = obs_shape[0] * self.prediction_horizon,
            )
            guide_base = None
        else:
            raise ValueError(f"Unknown diffusion model type: {args.diffusion_model_type}")

        # Create the guide and diffuser.
        self.guide = None if guide_base is None else OptimalityClassifier(guide_base).to(device)
        self.diffuser = DiscreteDiffusionSDE(
            diffuser_base,
            None,
            fix_mask,
            loss_weight,
            diffusion_steps=256,
            # classifier=self.guide,
            predict_noise=False
        ).to(device)

        # Update the diffuser optimizers.
        self.diffuser.manual_optimizers = {}
        self.diffuser.configure_manual_optimizers()


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None):
        ''' Get a prediction from the diffuser.
            Args:
                trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
                visibility_mask: A tensor of shape (T, D) indicating which parts of the observation are visible.
                prediction_prev: A tensor of shape (1, T, D) for autoregression.
            Returns:
                prediction: A tensor of shape (1, T, D) containing the predicted trajectory.
        '''

        # Set up trajectory and visibility mask.
        trajectory = trajectory.unsqueeze(0)  # Add sample dimension.
        if visibility_mask == None:
            visibility_mask = torch.ones_like(trajectory)  # Default to all visible.
        else:
            visibility_mask = visibility_mask.unsqueeze(0) # Add sample dimension.

        # Apply the visibility mask to the trajectory.
        trajectory = trajectory * visibility_mask

        # Autoregression
        if prediction_prev is not None and self.args.diffusion_autoregression_steps > 0:
            k = self.args.diffusion_autoregression_steps
            trajectory[:, :k] = torch.where(
                visibility_mask[:, :k] == 0,
                prediction_prev[:, -k:],
                trajectory[:, :k]
            )
            visibility_mask[:, :k] = 1

        # The trajectory and visibility_mask represent the known data and are applied as described by Janner et al.
        # We set the fix_mask manually here as a workaround for CleanDiffuser not taking it as an input.
        self.diffuser.fix_mask = torch.nn.Parameter(visibility_mask, requires_grad=False)

        # Sample from the diffusion model.
        with torch.enable_grad():
            prediction, log = self.diffuser.sample(
                prior=trajectory,
                solver="ddpm",
                n_samples=1,
                sample_steps=5,
                condition_cg=trajectory,
                condition_cg_mask=visibility_mask,
                w_cg=0.1,
                w_cfg=0.0
            )

        return prediction