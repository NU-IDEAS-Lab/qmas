"""Shared layout/offsets builder for the dense-GNN observation space.

The actor and the diffusion predictor must agree on the byte boundaries of the
flattened Dict observation produced when ``args.gnn_use_dense_obs=True``.
Centralizing the construction here guarantees both consumers see identical
offsets and makes it easy to add a runtime equality check.
"""

from typing import Dict, Tuple

import numpy as np


GRAPH_KEYS = (
    "node_features",
    "edge_index",
    "edge_attr",
    "agent_idx",
    "neighbors_mask",
    "num_nodes",
    "num_edges",
)


def _resolve_dict_space(obs_space):
    """Return the underlying gymnasium Dict space.

    The PettingZoo wrapper may flatten a Dict space into a Box and stash the
    original under ``_original_dict_space``; we fall back to ``obs_space`` if
    that attribute is absent (i.e., the caller already passed the Dict).
    """
    return getattr(obs_space, "_original_dict_space", None) or obs_space


def build_gnn_dense_offsets(obs_space) -> Tuple[Dict, Dict]:
    """Build (layout, offsets) for a dense-GNN observation Dict space.

    layout: dict with max_total_nodes / node_feat_dim / max_edges / edge_feat_dim.
    offsets: dict mapping subspace key -> (start, end, shape) into the flat vector.
    """
    dict_space = _resolve_dict_space(obs_space)

    nf_space = dict_space.spaces["node_features"]
    ea_space = dict_space.spaces["edge_attr"]
    max_total_nodes, node_feat_dim = nf_space.shape
    max_edges, edge_feat_dim = ea_space.shape

    layout = dict(
        max_total_nodes=int(max_total_nodes),
        node_feat_dim=int(node_feat_dim),
        edge_feat_dim=int(edge_feat_dim),
        max_edges=int(max_edges),
    )

    offsets: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
    cursor = 0
    for key, space in dict_space.spaces.items():
        size = int(np.prod(space.shape))
        offsets[key] = (cursor, cursor + size, tuple(space.shape))
        cursor += size

    layout["flat_dim"] = cursor
    return layout, offsets


def assert_offsets_equal(a: Dict, b: Dict) -> None:
    """Raise AssertionError if two offset dicts disagree.

    Used at startup to confirm the predictor and the actor see identical byte
    boundaries for the flattened Dict obs.
    """
    if set(a.keys()) != set(b.keys()):
        raise AssertionError(f"GNN dense offset key mismatch: {set(a)} vs {set(b)}")
    for key in a:
        if tuple(a[key]) != tuple(b[key]):
            raise AssertionError(f"GNN dense offset mismatch for {key!r}: {a[key]} vs {b[key]}")
