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

        if self.args.prediction_history_include_actions:
            transition_dim = obs_dim + action_dim
        else:
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
                depth=6,
                timestep_emb_type="fourier",
                timestep_emb_params={"scale": 0.1},
            )
            guide_base = HalfDiT1d(
                x_dim=transition_dim,
                out_dim=1,
                x_seq_len=self.prediction_horizon,
                emb_dim=128,
                d_model=256,
                n_heads=8,
                depth=4,
                timestep_emb_type="fourier",
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
            predict_noise=False,
            # ema_rate=0.9999,
            
        ).to(device)

        # Deregister fix_mask as nn.Parameter so per-batch assignments are plain
        # attribute sets (no _parameters bookkeeping, no EMA tracking).
        if "fix_mask" in self.diffuser._parameters:
            initial_fix_mask = self.diffuser._parameters.pop("fix_mask").data
            self.diffuser.fix_mask = initial_fix_mask

        # Update the diffuser optimizers.
        self.diffuser.manual_optimizers = {}
        self.diffuser.configure_manual_optimizers()

        # Create uncertainty bounds estimator if needed.
        self.uncertainty_bounds_estimator = None
        if args.prediction_uq_method == "estimation":
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
        # Running normalization statistics for predictor inputs.
        self.norm_eps = 1e-6
        self.register_buffer("obs_running_mean", torch.zeros_like(fix_mask, dtype=torch.float32))
        self.register_buffer("obs_running_var", torch.ones_like(fix_mask, dtype=torch.float32))
        self.register_buffer("obs_running_count", torch.full_like(fix_mask, self.norm_eps, dtype=torch.float32))

        if "diffusion" in self.diffuser.model:
            self.diffuser.model["diffusion"] = torch.compile(
                self.diffuser.model["diffusion"], fullgraph=False, mode="default"
            )
        if hasattr(self.diffuser, "model_ema") and "diffusion" in self.diffuser.model_ema:
            self.diffuser.model_ema["diffusion"] = torch.compile(
                self.diffuser.model_ema["diffusion"], fullgraph=False, mode="default"
            )


    def _ensure_norm_shape(self, x: torch.Tensor):
        """Ensure normalization buffers match current trajectory feature shape."""
        feature_shape = x.shape[1:]
        if tuple(self.obs_running_mean.shape) != tuple(feature_shape):
            self.obs_running_mean = torch.zeros(feature_shape, dtype=torch.float32, device=x.device)
            self.obs_running_var = torch.ones(feature_shape, dtype=torch.float32, device=x.device)
            self.obs_running_count = torch.full(feature_shape, self.norm_eps, dtype=torch.float32, device=x.device)


    @torch.no_grad()
    def update_normalization_stats(self, x: torch.Tensor, visibility_mask: torch.Tensor = None):
        """Update running mean/variance using visible trajectory entries only."""
        self._ensure_norm_shape(x)

        if visibility_mask is None:
            valid = torch.ones_like(x, dtype=torch.float32)
        else:
            valid = visibility_mask.to(dtype=torch.float32)

        x_float = x.to(dtype=torch.float32)
        raw_count = valid.sum(dim=0)
        has_obs = raw_count > 0
        safe_count = torch.where(has_obs, raw_count, torch.ones_like(raw_count))

        batch_mean = (x_float * valid).sum(dim=0) / safe_count
        centered = x_float - batch_mean.unsqueeze(0)
        batch_var = (centered.pow(2) * valid).sum(dim=0) / safe_count

        total = self.obs_running_count + raw_count
        delta = batch_mean - self.obs_running_mean

        new_mean = self.obs_running_mean + delta * raw_count / total.clamp_min(self.norm_eps)
        m_a = self.obs_running_var * self.obs_running_count
        m_b = batch_var * raw_count
        m2 = m_a + m_b + delta.pow(2) * self.obs_running_count * raw_count / total.clamp_min(self.norm_eps)
        new_var = m2 / total.clamp_min(self.norm_eps)

        self.obs_running_mean = torch.where(has_obs, new_mean, self.obs_running_mean)
        self.obs_running_var = torch.where(has_obs, new_var.clamp_min(self.norm_eps), self.obs_running_var)
        self.obs_running_count = torch.where(has_obs, total, self.obs_running_count)


    def normalize_trajectory(self, x: torch.Tensor):
        """Normalize trajectories using running statistics."""
        self._ensure_norm_shape(x)
        mean = self.obs_running_mean.to(device=x.device, dtype=x.dtype)
        std = torch.sqrt(self.obs_running_var.to(device=x.device, dtype=x.dtype).clamp_min(self.norm_eps))
        return (x - mean) / std


    def denormalize_trajectory(self, x: torch.Tensor):
        """Invert predictor trajectory normalization."""
        self._ensure_norm_shape(x)
        mean = self.obs_running_mean.to(device=x.device, dtype=x.dtype)
        std = torch.sqrt(self.obs_running_var.to(device=x.device, dtype=x.dtype).clamp_min(self.norm_eps))
        return x * std + mean


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

        # Normalize first, then apply visibility mask so unknown entries remain neutralized.
        trajectory = self.normalize_trajectory(trajectory)
        trajectory = trajectory * visibility_mask

        # Set up the warm start based on previous prediction if provided.
        warm_start_reference = None
        if prediction_prev is not None and self.args.prediction_warm_start:
            prediction_prev_normalized = self.normalize_trajectory(prediction_prev)
            warm_start_reference = torch.cat([prediction_prev_normalized[:, 1:], torch.randn_like(prediction_prev[:, :1])], dim=1)

        # Autoregression
        if prediction_prev is not None and self.args.diffusion_autoregression_steps > 0:
            k = self.args.diffusion_autoregression_steps
            prediction_prev = self.normalize_trajectory(prediction_prev)
            trajectory[:, :k] = torch.where(
                visibility_mask[:, :k] == 1,
                trajectory[:, :k],
                prediction_prev[:, -k:]
            )
            visibility_mask[:, :k] = 1

        if warm_start_reference is not None:
            warm_start_reference = torch.where(
                visibility_mask == 1,
                trajectory,
                warm_start_reference
            )

        # The trajectory and visibility_mask represent the known data and are applied as described by Janner et al.
        # We set the fix_mask manually here as a workaround for CleanDiffuser not taking it as an input.
        self.diffuser.fix_mask = visibility_mask

        # Sample from the diffusion model.
        autocast_device = "cuda" if trajectory.is_cuda else "cpu"
        with torch.enable_grad(), torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            prediction, log = self.diffuser.sample(
                prior=trajectory,
                solver="ddim",
                n_samples=trajectory.shape[0],
                temperature=0.6,
                sample_steps=20,
                condition_cg=trajectory,
                condition_cg_mask=visibility_mask,
                w_cg=0.0,
                w_cfg=0.0,
                warm_start_reference=warm_start_reference,
                # warm_start_forward_level=0.3,
                use_ema=True,
            )
        prediction = prediction.float()

        # Take the mean over the samples and map back to the original observation scale.
        # prediction = prediction.mean(dim=0, keepdim=True)
        prediction = self.denormalize_trajectory(prediction)

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
