import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from onpolicy.models.utils.util import check
from onpolicy.models.utils.cnn import CNNBase
from onpolicy.models.utils.mlp import MLPBase, MLPLayer
from onpolicy.models.utils.gnn import GNNBase
from onpolicy.models.utils.rnn import RNNLayer
from onpolicy.models.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space, get_graph_obs_space, strip_graph_obs_space, get_graph_obs_space_idx

from qmas.variant_two_modules.new.actor_critic import StateEncoder, QmasCritic
from qmas.variant_two_modules.new.predictor import Predictor


class EmbeddedPredictor(Predictor):
    """Predictor submodule whose parameters are owned by the enclosing actor_optimizer.

    The parent ``Predictor`` sets up internal Adam optimizers for the diffuser
    (and optionally for an uncertainty bounds estimator).  Those internal
    optimizers conflict with joint optimisation via the actor_optimizer, so
    this subclass clears them after construction.

    Training uses ``self.diffuser.loss(x0)`` to obtain a differentiable
    score-matching scalar that is added to the shared actor loss before the
    single backward pass.  After ``actor_optimizer.step()`` the caller must
    invoke ``self.diffuser.ema_update()`` to keep the EMA model current.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable internal diffuser optimizers — managed by actor_optimizer.
        self.diffuser.manual_optimizers = {}
        # Remove any standalone uncertainty optimizer created by the parent.
        if hasattr(self, 'uncertainty_optimizer'):
            del self.uncertainty_optimizer


class QmasActorOneModule(nn.Module):
    """Actor with an embedded diffusion predictor as a single differentiable model.

    Architecture:
    - **Local encoder** (MLP/GNN/CNN): encodes per-agent local observations.
    - **Prediction encoder** (``StateEncoder``): encodes the current global
      observation into a latent context vector concatenated with local features
      before the policy head.
    - **Diffusion predictor** (``EmbeddedPredictor``): a trajectory-level
      diffusion model (reused from ``variant_two_modules.new``) with its
      internal optimizers disabled.  Its parameters are registered as
      sub-module parameters and therefore covered by the single actor_optimizer.

    During training the algorithm computes:

        total_actor_loss = ppo_loss + prediction_loss_coef * diffuser.loss(trajectory)

    in a single ``backward()`` call so that both the PPO objective and the
    diffusion score-matching objective contribute to each parameter update.
    """

    def __init__(self, args, obs_space, share_obs_space, action_space,
                 device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_gnn = args.use_gnn_policy
        self._use_gnn_mlp = args.use_gnn_mlp_policy
        self._recurrent_N = args.recurrent_N
        self._use_rnn = self._use_naive_recurrent_policy or self._use_recurrent_policy
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.MAX_NEIGHBORS = 15
        self.MAX_NODES = 50

        self._use_cnn = False
        self._use_mlp = False
        self._use_attention = False

        # ------------------------------------------------------------------ #
        # Local observation encoder                                            #
        # ------------------------------------------------------------------ #
        if self._use_gnn:
            obs_space_graph = get_graph_obs_space(obs_space)
            obs_space_nongraph = strip_graph_obs_space(obs_space)
            self.obs_space_graph_idx = get_graph_obs_space_idx(obs_space)

            self.base = GNNBase(
                layers=args.gnn_layer_N,
                node_dim=get_shape_from_obs_space(obs_space_graph.node_space)[0],
                edge_dim=get_shape_from_obs_space(obs_space_graph.edge_space)[0],
                hidden_dim=args.gnn_hidden_size,
                output_dim=args.gnn_hidden_size,
                dropout_rate=args.gnn_dropout_rate,
                jk=args.gnn_skip_connections,
                use_orthogonal=args.use_orthogonal,
                use_ReLU=args.use_ReLU,
            )
            if args.gnn_neighbor_scoring:
                self.neighbor_scorer = MLPLayer(
                    input_dim=args.gnn_hidden_size, output_dim=1,
                    hidden_size=self.hidden_size, layer_N=3,
                    use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU,
                    use_layer_norm=False,
                )
                actor_base_out = self.MAX_NEIGHBORS + get_shape_from_obs_space(obs_space_nongraph)[0]
            else:
                actor_base_out = args.gnn_hidden_size + get_shape_from_obs_space(obs_space_nongraph)[0]

            if self._use_gnn_mlp:
                self.mlp0 = MLPLayer(
                    input_dim=actor_base_out, output_dim=self.hidden_size,
                    hidden_size=self.hidden_size, layer_N=args.layer_N,
                    use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU,
                )
                actor_base_out = self.hidden_size
        else:
            obs_shape = get_shape_from_obs_space(obs_space)
            if args.prediction_uq_injection_method == "append":
                obs_shape = (*obs_shape[:-1], obs_shape[-1] * 2)

            self._use_cnn = len(obs_shape) == 3
            self._use_mlp = True

            actor_base_in = int(np.prod(obs_shape))

            if self._use_cnn:
                self.cnn = CNNBase(args, obs_shape)
                actor_base_out = self.hidden_size
            if self._use_mlp:
                self.mlp = MLPBase(args, actor_base_in)
                actor_base_out = self.hidden_size

        print(
            f"QmasActorOneModule: use_gnn={self._use_gnn}, use_cnn={self._use_cnn}, "
            f"use_mlp={self._use_mlp}, use_rnn={self._use_rnn}"
        )

        # ------------------------------------------------------------------ #
        # Prediction encoder: encodes global_obs → latent context             #
        # Replaces both the separate StateEncoder and the standalone Predictor #
        # ------------------------------------------------------------------ #
        global_obs_shape = get_shape_from_obs_space(share_obs_space)
        if args.prediction_uq_injection_method == "append":
            global_obs_shape = (*global_obs_shape[:-1], global_obs_shape[-1] * 2)

        latent_dim = args.state_encoder_output_dim

        # ------------------------------------------------------------------ #
        # Prediction encoder: single-step global_obs → latent for policy     #
        # ------------------------------------------------------------------ #
        self.prediction_encoder = StateEncoder(
            args,
            global_obs_shape,
            output_dim=latent_dim,
            use_ReLU=args.use_ReLU,
            use_orthogonal=args.use_orthogonal,
            device=device,
        )

        actor_input_dim = actor_base_out + latent_dim

        # ------------------------------------------------------------------ #
        # Ensemble of diffusion predictor submodules                           #
        # All predictors live on the actor's device so gradients flow through  #
        # actor_optimizer.  Internal CleanDiffuser optimizers are disabled.    #
        # ------------------------------------------------------------------ #
        share_obs_shape_raw = get_shape_from_obs_space(share_obs_space, flatten_dicts=False)
        share_obs_dim = int(np.prod(share_obs_shape_raw))
        action_dim = int(np.prod(get_shape_from_act_space(action_space)))

        if args.prediction_ensemble_size > 1:
            print(f"QmasActorOneModule: Creating ensemble of {args.prediction_ensemble_size} predictors (all on {device}).")

        self.predictors = nn.ModuleList([
            EmbeddedPredictor(
                share_obs_dim,
                action_dim,
                args,
                obs_shape=share_obs_shape_raw,
                device=device,
            ).to(device)
            for _ in range(args.prediction_ensemble_size)
        ])

        # ------------------------------------------------------------------ #
        # RNN and action layers                                                #
        # ------------------------------------------------------------------ #
        if self._use_rnn:
            self.rnn = RNNLayer(actor_input_dim, self.hidden_size,
                                self._recurrent_N, self._use_orthogonal)
            actor_input_dim = self.hidden_size

        self.act = ACTLayer(action_space, actor_input_dim, self._use_orthogonal, self._gain)

        self.to(device)

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _encode_local_obs(self, obs) -> torch.Tensor:
        """Run the local observation through the base encoder."""
        if self._use_gnn:
            obs_graph = obs[:, self.obs_space_graph_idx]
            nonGraphIdx = [i for i in range(obs.shape[1]) if i != self.obs_space_graph_idx]
            obs_nongraph = obs[:, nonGraphIdx]
            if len(obs_nongraph.shape) > 1 and obs_nongraph.shape[1] > 0:
                obs_non_graph_float = np.zeros(
                    (obs_nongraph.shape[0], *obs_nongraph[0, 0].shape), dtype=np.float32
                )
                for i in range(obs_nongraph.shape[0]):
                    obs_non_graph_float[i] = obs_nongraph[i, 0]
                obs_nongraph = obs_non_graph_float
            obs_nongraph = check(obs_nongraph.astype(np.float32)).to(**self.tpdv)

            graphs = Batch.from_data_list(obs_graph).to(self.device, "x", "edge_attr", "edge_index")
            feats = self.base(graphs.x, graphs.edge_attr, graphs.edge_index)
            feats, _ = to_dense_batch(feats, graphs.batch.to(self.device))

            if hasattr(graphs, "neighbors") and self.args.gnn_neighbor_scoring:
                if self.MAX_NODES - feats.shape[1] > 0:
                    feats = F.pad(feats, (0, 0, 0, self.MAX_NODES - feats.shape[1]), value=0.0)
                neighbors_mask = check(np.array(graphs.neighbors_mask)).to(**self.tpdv).bool()
                neighbors_mask = neighbors_mask.unsqueeze(2).repeat(1, 1, feats.shape[-1])
                feats_masked = torch.where(neighbors_mask, feats, 0.0)
                scores = self.neighbor_scorer(feats_masked)
                scores_shifted = torch.zeros((feats.shape[0], self.MAX_NEIGHBORS), **self.tpdv)
                for i in range(feats.shape[0]):
                    nbrs = check(np.array(graphs.neighbors[i])).to(**self.tpdv).int()
                    scores_shifted[i, :nbrs.shape[0]] = scores[i, nbrs, 0]
                feats = scores_shifted
            elif hasattr(graphs, "agent_idx"):
                agent_idx = torch.from_numpy(np.array(graphs.agent_idx)).reshape(-1, 1).to(self.device)
                feats = self.base.gatherNodeFeats(feats, agent_idx)
            else:
                feats = self.base.graphAggr(feats, aggr="mean")

            feats = torch.cat([feats, obs_nongraph], dim=-1)
            if self._use_gnn_mlp:
                feats = self.mlp0(feats)
        else:
            obs = check(obs).to(**self.tpdv)
            feats = obs
            if self._use_cnn:
                feats = self.cnn(feats)
            if self._use_mlp:
                feats = self.mlp(feats)
        return feats

    def _encode_global_obs(self, actor_features: torch.Tensor, global_obs) -> torch.Tensor:
        """Encode global_obs via the prediction encoder and append to actor_features."""
        batch_size = actor_features.shape[0]
        latent_dim = self.args.state_encoder_output_dim

        if global_obs is None:
            latent = torch.zeros((batch_size, latent_dim), **self.tpdv)
        else:
            global_obs = check(global_obs).to(**self.tpdv)
            global_obs = torch.nan_to_num(global_obs, nan=0.0, posinf=0.0, neginf=0.0)
            latent = self.prediction_encoder(global_obs)
            if latent.shape[0] < batch_size:
                if batch_size % latent.shape[0] != 0:
                    raise ValueError(
                        "Batch size of obs is not a multiple of batch size of global_obs."
                    )
                latent = latent.repeat_interleave(batch_size // latent.shape[0], dim=0)

        return torch.cat([actor_features, latent], dim=-1)

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def forward(self, obs, rnn_states, masks, available_actions=None,
                deterministic=False, global_obs=None):
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self._encode_local_obs(obs)
        actor_features = self._encode_global_obs(actor_features, global_obs)

        if self._use_rnn:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, global_obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None):
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if global_obs is not None:
            global_obs = check(global_obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self._encode_local_obs(obs)
        actor_features = self._encode_global_obs(actor_features, global_obs)

        if self._use_rnn:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )
        return action_log_probs, dist_entropy
