"""GNN backbone for the diffusion predictor.

Re-interprets the flat ``(B, T, D_flat)`` trajectory the buffer already produces
as a per-timestep graph by slicing through the dense GNN obs layout, applies a
SparseDiff-style graph transformer per timestep, mixes across the temporal
axis, then re-packs the node-feature region back into the flat shape.

Edge slots in ``x`` are preserved bytewise — only node-feature slots are
written. ``Predictor.fix_mask`` must pin all non-node-feature regions to 1 so
the diffusion noise schedule does not corrupt them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Make the vendored SparseDiff package importable. The submodule lives at
# ``third_party/SparseDiff/sparse_diffusion`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPARSEDIFF_ROOT = _REPO_ROOT / "third_party" / "SparseDiff"
if str(_SPARSEDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARSEDIFF_ROOT))

# ``sparse_diffusion.utils`` pulls in omegaconf / hydra / lightning, which we
# do not need for the denoiser layers we use. Stub it before SparseDiff loads
# so the module-level ``from sparse_diffusion import utils`` in
# ``conv_transformer_model.py`` resolves to a harmless empty module. Only
# ``GraphTransformerConv`` (which we do not use) actually touches utils.
import types as _types  # noqa: E402

if "sparse_diffusion.utils" not in sys.modules:
    _stub = _types.ModuleType("sparse_diffusion.utils")
    # ``GraphTransformerConv`` references these as type annotations at class
    # definition time. We never use that class, but the references must
    # resolve. Provide trivial stand-ins so the module loads.
    _stub.PlaceHolder = type("PlaceHolder", (), {})
    _stub.SparsePlaceHolder = type("SparsePlaceHolder", (), {})
    sys.modules["sparse_diffusion.utils"] = _stub
    # Ensure the parent package exists too, so ``from sparse_diffusion import utils``
    # succeeds without executing sparse_diffusion/__init__.py side effects.
    if "sparse_diffusion" not in sys.modules:
        _pkg = _types.ModuleType("sparse_diffusion")
        _pkg.__path__ = [str(_SPARSEDIFF_ROOT / "sparse_diffusion")]
        sys.modules["sparse_diffusion"] = _pkg
    setattr(sys.modules["sparse_diffusion"], "utils", _stub)

from sparse_diffusion.models.conv_transformer_model import XEyTransformerLayer  # noqa: E402

from cleandiffuser.nn_diffusion.base_nn_diffusion import BaseNNDiffusion  # noqa: E402


class GnnDiffusion1d(BaseNNDiffusion):
    """Per-timestep graph-transformer denoiser with a temporal mixer.

    Forward IO matches DiT1d's contract: ``x`` is ``(B, T, D_flat)``, ``t`` is
    ``(B,)``, output is ``(B, T, D_flat)``. All non-node-feature slots in
    ``x`` are passed through unchanged.
    """

    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        layout: Dict,
        offsets: Dict,
        emb_dim: int = 128,
        d_model: int = 128,
        d_edge: int = 32,
        d_y: int = 64,
        depth: int = 4,
        n_heads: int = 4,
        temporal_depth: int = 2,
        temporal_n_heads: int = 4,
        dropout: float = 0.0,
        timestep_emb_type: str = "fourier",
        timestep_emb_params: Optional[Dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.x_dim = int(x_dim)
        self.x_seq_len = int(x_seq_len)
        self.layout = layout
        self.offsets = offsets

        N = layout["max_total_nodes"]
        F = layout["node_feat_dim"]
        E = layout["max_edges"]
        EF = layout["edge_feat_dim"]
        self._N = N
        self._F = F
        self._E = E
        self._EF = EF

        # Slice indices into the flat ``(T, D_flat)`` per-step vector.
        self._nf_start, self._nf_end, _ = offsets["node_features"]
        self._ei_start, self._ei_end, _ = offsets["edge_index"]
        self._ea_start, self._ea_end, _ = offsets["edge_attr"]
        self._nn_start, self._nn_end, _ = offsets["num_nodes"]
        self._ne_start, self._ne_end, _ = offsets["num_edges"]

        if self._nf_end - self._nf_start != N * F:
            raise ValueError("layout/offsets disagree on node_features size")
        if self._ea_end - self._ea_start != E * EF:
            raise ValueError("layout/offsets disagree on edge_attr size")

        # Input/output projections. The graph layers operate in a hidden space.
        self.lin_in_X = nn.Linear(F, d_model)
        self.lin_in_E = nn.Linear(EF, d_edge)
        # The y stream carries the diffusion timestep embedding (broadcast per
        # graph) plus optional per-graph context. Build it from the timestep
        # embedding only for now.
        self.lin_in_y = nn.Linear(emb_dim, d_y)

        # Stack SparseDiff-style XEy transformer layers. Use last_layer=True on
        # every layer so the y stream gets updated at each step (cheap; depth
        # is small).
        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=d_model,
                    de=d_edge,
                    dy=d_y,
                    n_head=n_heads,
                    dim_ffX=2 * d_model,
                    dim_ffE=2 * d_edge,
                    dim_ffy=2 * d_y,
                    dropout=dropout,
                    last_layer=True,
                )
                for _ in range(depth)
            ]
        )
        self.out_ln_X = nn.LayerNorm(d_model)
        self.lin_out_X = nn.Linear(d_model, F)

        # Temporal mixer: shared-weights TransformerEncoder over the T axis
        # applied per node slot. Operates in d_model space — keeping it off
        # the raw F-dim space avoids embed_dim/num_heads divisibility
        # constraints when F is small.
        if temporal_depth > 0:
            if d_model % temporal_n_heads != 0:
                raise ValueError(
                    f"gnn_diffusion_d_model ({d_model}) must be divisible by "
                    f"gnn_diffusion_temporal_n_heads ({temporal_n_heads})."
                )
            self.temporal_pos = nn.Parameter(torch.zeros(1, x_seq_len, d_model))
            nn.init.normal_(self.temporal_pos, std=0.02)
            tlayer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=temporal_n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_mixer = nn.TransformerEncoder(tlayer, num_layers=temporal_depth)
        else:
            self.temporal_pos = None
            self.temporal_mixer = None

    # ------------------------------------------------------------------
    # Split / pack helpers
    # ------------------------------------------------------------------

    def _split_flat(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split ``(B, T, D_flat)`` into structured per-step graph tensors."""
        B, T, D = x.shape
        assert D == self.x_dim, f"expected D_flat={self.x_dim}, got {D}"
        node_feat = x[..., self._nf_start:self._nf_end].reshape(B, T, self._N, self._F)
        edge_index = x[..., self._ei_start:self._ei_end].reshape(B, T, 2, self._E)
        edge_attr = x[..., self._ea_start:self._ea_end].reshape(B, T, self._E, self._EF)
        num_nodes = x[..., self._nn_start:self._nn_end].reshape(B, T)
        num_edges = x[..., self._ne_start:self._ne_end].reshape(B, T)
        return node_feat, edge_index, edge_attr, num_nodes, num_edges

    def _pack_node_features(self, x_in: torch.Tensor, new_node_feat: torch.Tensor) -> torch.Tensor:
        """Return ``x_in`` with the node_features region replaced.

        All other slots (edge_index, edge_attr, num_nodes, num_edges, agent_idx,
        neighbors_mask, ...) are bytewise unchanged.
        """
        B, T, D = x_in.shape
        out = x_in.clone()
        out[..., self._nf_start:self._nf_end] = new_node_feat.reshape(B, T, -1)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, T, D_flat) -> (B, T, D_flat)``.

        Only node-feature slots are written. Per-step adjacency comes from
        ``edge_index`` / ``edge_attr`` slices and is treated as conditioning.
        """
        B, T, D = x.shape
        N, Fdim, E = self._N, self._F, self._E
        device = x.device

        node_feat, edge_index_f, edge_attr, num_nodes_f, num_edges_f = self._split_flat(x)

        # Validity masks. ``num_nodes`` / ``num_edges`` are stored as float32 in
        # the obs space; round and clamp to integer counts.
        num_nodes = num_nodes_f.round().clamp(min=0, max=N).long()  # (B, T)
        num_edges = num_edges_f.round().clamp(min=0, max=E).long()  # (B, T)
        node_range = torch.arange(N, device=device)
        edge_range = torch.arange(E, device=device)
        node_mask = node_range.view(1, 1, N) < num_nodes.unsqueeze(-1)  # (B, T, N)
        edge_mask = edge_range.view(1, 1, E) < num_edges.unsqueeze(-1)  # (B, T, E)

        # Project to hidden dims. Mask invalid slots to zero so they do not
        # contribute through the linear layers' biases.
        x_h = self.lin_in_X(node_feat) * node_mask.unsqueeze(-1).to(node_feat.dtype)
        e_h = self.lin_in_E(edge_attr) * edge_mask.unsqueeze(-1).to(edge_attr.dtype)

        # Build the sparse batch over B*T graphs.
        # Cumulative offsets so each graph's edge_index references its node
        # range inside the flat node tensor.
        node_counts = num_nodes.reshape(-1)  # (B*T,)
        edge_counts = num_edges.reshape(-1)  # (B*T,)
        n_graphs = B * T
        cum_nodes = torch.zeros(n_graphs, dtype=torch.long, device=device)
        cum_nodes[1:] = node_counts[:-1].cumsum(0)

        # Flatten nodes: gather only valid slots.
        x_flat_nodes = x_h.reshape(B * T, N, -1)[node_mask.reshape(B * T, N)]  # (sum_N, d_model)
        # Per-node graph index for the y FiLM gather inside TransformerConv.
        batch_vec = torch.repeat_interleave(
            torch.arange(n_graphs, device=device), node_counts
        )  # (sum_N,)

        # Flatten edges and remap edge_index to global node indices.
        ei = edge_index_f.long().clamp(min=0, max=N - 1)  # (B, T, 2, E)
        ei = ei.reshape(B * T, 2, E)
        edge_offsets = cum_nodes.view(B * T, 1, 1).to(ei.dtype)
        ei_global = ei + edge_offsets
        # Mask out invalid edge slots.
        em = edge_mask.reshape(B * T, E)
        # Gather valid edges across all graphs.
        flat_em = em.reshape(-1)
        ei_global_flat = ei_global.permute(1, 0, 2).reshape(2, -1)  # (2, B*T*E)
        ea_h_flat = e_h.reshape(B * T * E, -1)
        edge_index_flat = ei_global_flat[:, flat_em]  # (2, sum_M)
        edge_attr_flat = ea_h_flat[flat_em]  # (sum_M, d_edge)

        # Build y from the diffusion timestep embedding. ``noise`` is shape
        # (B,); broadcast across T so each (b, t) graph carries the same
        # embedding for that batch element.
        noise_emb = self.map_noise(noise)  # (B, emb_dim)
        noise_emb = noise_emb.unsqueeze(1).expand(B, T, -1).reshape(n_graphs, -1)
        y = self.lin_in_y(noise_emb)  # (n_graphs, d_y)

        # If sum_N is 0 we have nothing to denoise; bail out by returning x
        # unchanged. (Should not happen in practice — at least some nodes
        # exist in every observation.)
        if x_flat_nodes.shape[0] == 0:
            return x

        # Apply graph-transformer layers.
        for layer in self.tf_layers:
            x_flat_nodes, edge_attr_flat, y = layer(
                x_flat_nodes, edge_index_flat, edge_attr_flat, y, batch_vec
            )

        # Scatter the d_model-dim hidden features back into (B, T, N, d_model);
        # padded slots stay 0. The temporal mixer runs in this hidden space.
        d_model = x_flat_nodes.shape[-1]
        hidden = node_feat.new_zeros(B, T, N, d_model)
        hidden.reshape(B * T, N, d_model)[node_mask.reshape(B * T, N)] = x_flat_nodes

        # Temporal mixer over the T axis, per-node, shared weights.
        if self.temporal_mixer is not None:
            # (B, T, N, d_model) -> (B*N, T, d_model)
            tmp = hidden.permute(0, 2, 1, 3).reshape(B * N, T, d_model)
            tmp = tmp + self.temporal_pos
            # Mask out padded nodes per (b, t) — a slot may be valid at some t
            # and invalid at others. Key-padding mask shape (B*N, T).
            kpm = (~node_mask.permute(0, 2, 1).reshape(B * N, T)).contiguous()
            # If a row is fully padded, key_padding_mask=all-True triggers NaNs
            # in attention softmax; force at least one valid position.
            all_pad = kpm.all(dim=1, keepdim=True)
            kpm = kpm & ~all_pad
            tmp = self.temporal_mixer(tmp, src_key_padding_mask=kpm)
            hidden = tmp.reshape(B, N, T, d_model).permute(0, 2, 1, 3).contiguous()
            # Re-zero padded slots so they do not contaminate the output.
            hidden = hidden * node_mask.unsqueeze(-1).to(hidden.dtype)

        # Project back to F-dim per node at the output.
        new_node_feat = self.lin_out_X(self.out_ln_X(hidden))
        new_node_feat = new_node_feat * node_mask.unsqueeze(-1).to(new_node_feat.dtype)

        return self._pack_node_features(x, new_node_feat)
