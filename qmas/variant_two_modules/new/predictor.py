import torch

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.classifier import OptimalityClassifier
from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d, UNet2d

from onpolicy.models.utils.mlp import MLPLayer


class UncertaintyBoundsEstimator(torch.nn.Module):
    ''' This class learns estimated bounds on independent variables. '''

    def __init__(self, input_dim, hidden_size, layer_N, use_ReLU, use_orthogonal, device=torch.device("cpu"), gain=None):
        super(UncertaintyBoundsEstimator, self).__init__()
        self.device = device

        # Output two values per input element (lower and upper bounds), so output_dim = 2 * input_dim
        self.input_dim = input_dim
        self.mlp = MLPLayer(input_dim=input_dim, output_dim=2 * input_dim, hidden_size=hidden_size, layer_N=layer_N, use_orthogonal=use_orthogonal, use_ReLU=use_ReLU, gain=gain)

        self.to(device)
    
    def forward(self, x):
        x = self.mlp(x)
        # Split the output into lower and upper bounds for every input element.
        lower_bound = x[:, :self.input_dim]
        upper_bound = x[:, self.input_dim:2 * self.input_dim]
        return lower_bound, upper_bound

    def loss(self, predicted_value, predicted_lower_bound, predicted_upper_bound):
        ''' Calculate the loss for the uncertainty bounds estimator.
            Args:
                predicted_value: A tensor of shape (batch_size, ...) containing the predicted values.
                predicted_lower_bound: A tensor of shape (batch_size, ...) containing the predicted lower bound.
                predicted_upper_bound: A tensor of shape (batch_size, ...) containing the predicted upper bound.
            Returns:
                loss: A tensor containing the loss value.
        '''

        # Hinge loss to ensure predicted_value is within bounds.
        zero_tensor = torch.zeros_like(predicted_value)
        lower_hinge = torch.max(predicted_lower_bound - predicted_value, zero_tensor)
        upper_hinge = torch.max(predicted_value - predicted_upper_bound, zero_tensor)
        loss = torch.mean(lower_hinge ** 2 + upper_hinge ** 2)
        return loss


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
            classifier=self.guide,
            predict_noise=False
        ).to(device)

        # Update the diffuser optimizers.
        self.diffuser.manual_optimizers = {}
        self.diffuser.configure_manual_optimizers()

        # Create uncertainty bounds estimator if needed.
        self.uncertainty_bounds_estimator = None
        if args.prediction_estimate_uncertainty:
            print("Creating uncertainty bounds estimator for predictor.")
            self.uncertainty_bounds_estimator = UncertaintyBoundsEstimator(
                input_dim=self.prediction_horizon * transition_dim,
                hidden_size=args.hidden_size,
                layer_N=args.layer_N,
                use_ReLU=args.use_ReLU,
                use_orthogonal=args.use_orthogonal,
                device=device
            )
            # Create an optimizer for the uncertainty bounds estimator.
            lr = args.lr
            self.uncertainty_optimizer = torch.optim.Adam(self.uncertainty_bounds_estimator.parameters(), lr=lr)


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None, has_sample_dim=False):
        ''' Get a prediction from the diffuser.
            Args:
                trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
                visibility_mask: A tensor of shape (T, D) indicating which parts of the observation are visible.
                prediction_prev: A tensor of shape (1, T, D) for autoregression.
            Returns:
                prediction: A tensor of shape (1, T, D) containing the predicted trajectory.
        '''

        # Set up trajectory and visibility mask.
        if not has_sample_dim:
            trajectory = trajectory.unsqueeze(0)  # Add sample dimension.
        if visibility_mask == None:
            visibility_mask = torch.ones_like(trajectory)  # Default to all visible.
        elif not has_sample_dim:
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


    def get_uncertainty_bounds(self, trajectory):
        ''' Get uncertainty bounds from the uncertainty bounds estimator.
            Args:
                trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
            Returns:
                lower_bound: A tensor of shape (T, 1) containing the predicted lower bound.
                upper_bound: A tensor of shape (T, 1) containing the predicted upper bound.
        '''
        assert self.uncertainty_bounds_estimator is not None, "Uncertainty bounds estimator is not defined."
        # trajectory_flat = trajectory.view(trajectory.shape[0], -1)  # Flatten the trajectory.
        # lower_bound, upper_bound = self.uncertainty_bounds_estimator(trajectory_flat)
        lower_bound, upper_bound = self.uncertainty_bounds_estimator(trajectory.flatten(start_dim=1))
        return lower_bound, upper_bound
