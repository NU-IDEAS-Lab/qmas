import torch

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.classifier import OptimalityClassifier
from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d, UNet2d

from onpolicy.models.utils.mlp import MLPLayer

from qmas.variant_two_modules.new.gnn_layout import build_gnn_dense_offsets


# Slot keys (in the dense GNN obs Dict) whose contents must not be modified by
# the diffusion process — adding Gaussian noise to integer indices, counts, or
# binary masks scrambles the graph structure. The GNN backbone passes them
# through unchanged in its forward; ``fix_mask`` pins them through the
# noise schedule in training and sampling.
_GNN_PINNED_KEYS = (
    "edge_index",
    "edge_attr",
    "num_nodes",
    "num_edges",
    "agent_idx",
    "neighbors_mask",
)


def compute_gnn_edge_fix_mask(offsets, T: int, D_flat: int) -> torch.Tensor:
    """Build a (T, D_flat) mask with 1 on every pinned slot, 0 on node features.

    The mask is OR-merged into per-batch visibility before assigning to
    ``self.diffuser.fix_mask`` so that edge structure / counts / agent
    positions are bytewise preserved through both forward diffusion (training)
    and reverse sampling.
    """
    mask = torch.zeros((T, D_flat), dtype=torch.float32)
    for key in _GNN_PINNED_KEYS:
        if key not in offsets:
            continue
        s, e, _ = offsets[key]
        mask[:, s:e] = 1.0
    return mask


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
    def __init__(self, obs_dim, action_dim, args, device=None, obs_shape=None,
                 obs_dict_space=None):
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

        # Set up GNN-specific bookkeeping (used only by the "gnn" backbone).
        self._gnn_layout = None
        self._gnn_offsets = None
        self._gnn_edge_fix_template = None  # (T, D_flat) buffer registered below if used
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
        elif args.diffusion_model_type == "gnn":
            if obs_dict_space is None:
                raise ValueError(
                    "diffusion_model_type='gnn' requires obs_dict_space (the "
                    "dense-graph Dict obs space) to be passed to Predictor()."
                )
            # Local import keeps SparseDiff-related work out of the import path
            # for non-GNN runs.
            from qmas.variant_two_modules.new.gnn_diffusion import GnnDiffusion1d

            self._gnn_layout, self._gnn_offsets = build_gnn_dense_offsets(obs_dict_space)
            if self.args.prediction_history_include_actions:
                raise ValueError(
                    "GNN diffusion backbone is incompatible with "
                    "prediction_history_include_actions=True (the action slots "
                    "would not align with the GNN obs layout). Set the flag to "
                    "False for the gnn backbone."
                )
            if self._gnn_layout["flat_dim"] != transition_dim:
                raise ValueError(
                    f"GNN obs flat_dim ({self._gnn_layout['flat_dim']}) does not "
                    f"match Predictor transition_dim ({transition_dim}); the "
                    "obs space and predictor inputs are out of sync."
                )

            diffuser_base = GnnDiffusion1d(
                x_dim=transition_dim,
                x_seq_len=self.prediction_horizon,
                layout=self._gnn_layout,
                offsets=self._gnn_offsets,
                emb_dim=128,
                d_model=args.gnn_diffusion_d_model,
                d_edge=args.gnn_diffusion_d_edge,
                d_y=args.gnn_diffusion_d_y,
                depth=args.gnn_diffusion_depth,
                n_heads=args.gnn_diffusion_n_heads,
                temporal_depth=args.gnn_diffusion_temporal_depth,
                temporal_n_heads=args.gnn_diffusion_temporal_n_heads,
                timestep_emb_type="fourier",
                timestep_emb_params={"scale": 0.1},
            )
            guide_base = None  # classifier guidance for the GNN path is out of scope for now.

            # Pin every non-node-feature slot through the noise schedule.
            self._gnn_edge_fix_template = compute_gnn_edge_fix_mask(
                self._gnn_offsets, self.prediction_horizon, transition_dim
            )
            # Loss weight: zero on pinned slots so the regression loss only
            # scores the node-feature region the backbone actually writes.
            loss_weight = (1.0 - self._gnn_edge_fix_template).clone()
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
            ema_rate=0.995,
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
        # Tracks whether the running stats have been one-shot seeded from a large batch.
        # Persisted as a buffer so it survives checkpoint save/load.
        self.register_buffer("normalizer_seeded", torch.tensor(False))

        # Edge-pin fix_mask template for the GNN backbone. Registered as a
        # buffer so .to(device) moves it; OR-merged into per-batch visibility
        # at training and sampling time.
        if self._gnn_edge_fix_template is not None:
            self.register_buffer(
                "gnn_edge_fix_template",
                self._gnn_edge_fix_template.to(dtype=torch.float32),
            )

        # Per-feature normalization stats for the GNN backbone. Shape (F,) so
        # the same (mean, var) applies to every (T, N_max) node slot — this is
        # what makes normalization invariant to per-episode node reordering.
        # Edge attrs and structural slots stay passthrough (handled via the
        # edge-pin template), so we only track node-feature stats here.
        if self._gnn_layout is not None:
            F = self._gnn_layout["node_feat_dim"]
            self.register_buffer("gnn_node_running_mean", torch.zeros(F, dtype=torch.float32))
            self.register_buffer("gnn_node_running_var", torch.ones(F, dtype=torch.float32))
            self.register_buffer("gnn_node_running_count", torch.tensor(0.0, dtype=torch.float32))


    def _ensure_norm_shape(self, x: torch.Tensor):
        """Ensure normalization buffers match current trajectory feature shape."""
        feature_shape = x.shape[1:]
        if tuple(self.obs_running_mean.shape) != tuple(feature_shape):
            self.obs_running_mean = torch.zeros(feature_shape, dtype=torch.float32, device=x.device)
            self.obs_running_var = torch.ones(feature_shape, dtype=torch.float32, device=x.device)
            self.obs_running_count = torch.full(feature_shape, self.norm_eps, dtype=torch.float32, device=x.device)


    @torch.no_grad()
    def seed_normalization_stats(self, x: torch.Tensor, visibility_mask: torch.Tensor = None):
        """One-shot initialization of running mean/var from a large batch.

        Why: incremental Welford updates take many minibatches to converge from the
        zero-mean / unit-var prior, so the diffuser chases a moving normalization
        target during early training. Seeding once with a large sample sets
        `obs_running_count` high enough that subsequent incremental updates barely
        move the stats, giving the diffuser stationary normalized inputs.
        """
        if self._gnn_layout is not None:
            self._gnn_seed_norm(x)
            return

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

        self.obs_running_mean = torch.where(has_obs, batch_mean, self.obs_running_mean)
        self.obs_running_var = torch.where(has_obs, batch_var.clamp_min(self.norm_eps), self.obs_running_var)
        self.obs_running_count = torch.where(has_obs, raw_count, self.obs_running_count)
        self.normalizer_seeded.fill_(True)


    @torch.no_grad()
    def update_normalization_stats(self, x: torch.Tensor, visibility_mask: torch.Tensor = None):
        """Update running mean/variance using visible trajectory entries only."""
        if self._gnn_layout is not None:
            self._gnn_update_norm(x)
            return

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


    def _gnn_passthrough_mask(self, x: torch.Tensor):
        """Return a (T, D_flat) mask that is 1 on pinned slots, else 0.

        For the legacy per-slot normalize path, structural slots must round-trip
        bytewise. Unused on the GNN per-feature path because that path only
        touches the node-feature region.
        """
        if self._gnn_edge_fix_template is None:
            return None
        return self.gnn_edge_fix_template.to(device=x.device, dtype=x.dtype)


    # ------------------------------------------------------------------
    # GNN per-feature normalization helpers
    #
    # Stats are shape (F,) — one (mean, var) per node-feature dim — and are
    # broadcast across all (T, N_max) node slots. This is what makes the
    # predictor's normalized inputs invariant to per-episode node reordering:
    # the same physical node landing in slot 47 vs slot 12 across episodes
    # gets the same normalization, because the stats key on the feature, not
    # the slot.
    #
    # Aggregation only counts *valid* node slots, gated by the per-step
    # ``num_nodes`` field embedded in ``x`` itself. Padded slots and edge /
    # structural slots are not touched.
    # ------------------------------------------------------------------

    def _gnn_node_view(self, x: torch.Tensor):
        """Return a view onto the node-feature region as ``(..., N_max, F)``."""
        nf_s, nf_e, _ = self._gnn_offsets["node_features"]
        N = self._gnn_layout["max_total_nodes"]
        F = self._gnn_layout["node_feat_dim"]
        nf = x[..., nf_s:nf_e]
        return nf.reshape(*nf.shape[:-1], N, F), (nf_s, nf_e, N, F)

    def _gnn_node_valid_mask(self, x: torch.Tensor):
        """Return a (..., N_max) bool mask derived from the ``num_nodes`` slot.

        Padded node slots are excluded from normalization stats (and from
        normalization itself); the pad-zero values shouldn't poison per-feature
        stats just because the env's max graph size exceeds the actual graph.
        """
        nn_s, nn_e, _ = self._gnn_offsets["num_nodes"]
        N = self._gnn_layout["max_total_nodes"]
        num_nodes = x[..., nn_s:nn_e].reshape(*x.shape[:-1])
        num_nodes = num_nodes.round().clamp(min=0, max=N).long()
        node_range = torch.arange(N, device=x.device)
        # Broadcast-friendly view: (..., 1) < (..., N_max) -> (..., N_max).
        return node_range < num_nodes.unsqueeze(-1)

    def _gnn_normalize_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-feature normalization to the node-feature region only."""
        out = x.clone()
        nf, (nf_s, nf_e, N, F) = self._gnn_node_view(x)
        valid = self._gnn_node_valid_mask(x).to(dtype=nf.dtype)
        mean = self.gnn_node_running_mean.to(device=x.device, dtype=x.dtype)
        std = torch.sqrt(
            self.gnn_node_running_var.to(device=x.device, dtype=x.dtype).clamp_min(self.norm_eps)
        )
        normed = (nf - mean) / std
        # Padded slots stay 0 — the GNN backbone masks them via num_nodes
        # anyway, and zero is the same value the env emits, so the
        # round-trip through normalize/denormalize stays bytewise consistent.
        normed = normed * valid.unsqueeze(-1)
        out[..., nf_s:nf_e] = normed.reshape(*normed.shape[:-2], -1)
        return out

    def _gnn_denormalize_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_gnn_normalize_node_features`."""
        out = x.clone()
        nf, (nf_s, nf_e, N, F) = self._gnn_node_view(x)
        valid = self._gnn_node_valid_mask(x).to(dtype=nf.dtype)
        mean = self.gnn_node_running_mean.to(device=x.device, dtype=x.dtype)
        std = torch.sqrt(
            self.gnn_node_running_var.to(device=x.device, dtype=x.dtype).clamp_min(self.norm_eps)
        )
        denormed = nf * std + mean
        denormed = denormed * valid.unsqueeze(-1)
        out[..., nf_s:nf_e] = denormed.reshape(*denormed.shape[:-2], -1)
        return out

    @torch.no_grad()
    def _gnn_aggregate_node_stats(self, x: torch.Tensor):
        """Return (batch_mean, batch_var, raw_count) over valid node slots in x.

        Shapes: ``(F,), (F,), scalar``.
        """
        nf, (_, _, N, F) = self._gnn_node_view(x)
        valid = self._gnn_node_valid_mask(x).to(dtype=nf.dtype)
        nf_flat = nf.reshape(-1, F).to(dtype=torch.float32)
        weight = valid.reshape(-1).to(dtype=torch.float32).unsqueeze(-1)  # (BTN, 1)
        raw_count = weight.sum()
        safe_count = raw_count.clamp_min(1.0)
        batch_mean = (nf_flat * weight).sum(0) / safe_count
        centered = nf_flat - batch_mean.unsqueeze(0)
        batch_var = (centered.pow(2) * weight).sum(0) / safe_count
        return batch_mean, batch_var, raw_count

    @torch.no_grad()
    def _gnn_seed_norm(self, x: torch.Tensor):
        batch_mean, batch_var, raw_count = self._gnn_aggregate_node_stats(x)
        if raw_count.item() == 0:
            return
        self.gnn_node_running_mean = batch_mean.to(self.gnn_node_running_mean.dtype)
        self.gnn_node_running_var = batch_var.clamp_min(self.norm_eps).to(
            self.gnn_node_running_var.dtype
        )
        self.gnn_node_running_count = raw_count.to(self.gnn_node_running_count.dtype)
        self.normalizer_seeded.fill_(True)

    @torch.no_grad()
    def _gnn_update_norm(self, x: torch.Tensor):
        batch_mean, batch_var, raw_count = self._gnn_aggregate_node_stats(x)
        if raw_count.item() == 0:
            return
        prev_count = self.gnn_node_running_count
        total = prev_count + raw_count
        delta = batch_mean - self.gnn_node_running_mean
        new_mean = self.gnn_node_running_mean + delta * raw_count / total.clamp_min(self.norm_eps)
        m_a = self.gnn_node_running_var * prev_count
        m_b = batch_var * raw_count
        m2 = (
            m_a + m_b + delta.pow(2) * prev_count * raw_count / total.clamp_min(self.norm_eps)
        )
        new_var = m2 / total.clamp_min(self.norm_eps)
        self.gnn_node_running_mean.copy_(new_mean)
        self.gnn_node_running_var.copy_(new_var.clamp_min(self.norm_eps))
        self.gnn_node_running_count.copy_(total)

    def normalize_trajectory(self, x: torch.Tensor):
        """Normalize trajectories using running statistics.

        Dispatches to the GNN per-feature path when the GNN backbone is in
        use; otherwise applies the legacy per-(t, slot) z-score.
        """
        if self._gnn_layout is not None:
            return self._gnn_normalize_node_features(x)

        self._ensure_norm_shape(x)
        mean = self.obs_running_mean.to(device=x.device, dtype=x.dtype)
        std = torch.sqrt(self.obs_running_var.to(device=x.device, dtype=x.dtype).clamp_min(self.norm_eps))
        normed = (x - mean) / std
        passthrough = self._gnn_passthrough_mask(x)
        if passthrough is not None:
            normed = torch.where(passthrough.bool(), x, normed)
        return normed


    def denormalize_trajectory(self, x: torch.Tensor):
        """Invert predictor trajectory normalization."""
        if self._gnn_layout is not None:
            return self._gnn_denormalize_node_features(x)

        self._ensure_norm_shape(x)
        mean = self.obs_running_mean.to(device=x.device, dtype=x.dtype)
        std = torch.sqrt(self.obs_running_var.to(device=x.device, dtype=x.dtype).clamp_min(self.norm_eps))
        out = x * std + mean
        passthrough = self._gnn_passthrough_mask(x)
        if passthrough is not None:
            out = torch.where(passthrough.bool(), x, out)
        return out


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
        # For the GNN backbone, OR-merge the edge-pin template so adjacency
        # / counts / agent positions are preserved through the noise schedule.
        if self._gnn_edge_fix_template is not None:
            edge_pin = self.gnn_edge_fix_template.to(
                device=visibility_mask.device, dtype=visibility_mask.dtype
            )
            visibility_mask = torch.maximum(visibility_mask, edge_pin)
        self.diffuser.fix_mask = visibility_mask

        # Sample from the diffusion model.
        needs_grad = (self.guide is not None) and (getattr(self.args, "diffusion_w_cg", 0.0) != 0.0)
        ctx = torch.enable_grad() if needs_grad else torch.no_grad()
        with ctx:
            prediction, log = self.diffuser.sample(
                prior=trajectory,
                solver="ddim",
                n_samples=trajectory.shape[0],
                temperature=0.0,
                sample_steps=self.args.diffusion_sample_steps,
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
