import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium.spaces as spaces
import numpy as np

from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from onpolicy.models.utils.util import init, check
from onpolicy.models.utils.cnn import CNNBase
from onpolicy.models.utils.mlp import MLPBase, MLPLayer
from onpolicy.models.utils.gnn import GNNBase
from onpolicy.models.utils.rnn import RNNLayer
from onpolicy.models.utils.act import ACTLayer
from onpolicy.models.utils.attention import SelfAttention
from onpolicy.utils.util import get_shape_from_obs_space, get_graph_obs_space, strip_graph_obs_space, get_graph_obs_space_idx, has_graph_obs_space
from onpolicy.models.r_actor_critic import R_Actor as Actor, R_Critic as Critic

class StateEncoder(nn.Module):
    ''' This class encodes the global state for communication purposes. '''

    def __init__(self, args, obs_shape, output_dim, use_ReLU, use_orthogonal, device=torch.device("cpu"), gain=None):
        super(StateEncoder, self).__init__()
        self.device = device
        self.args = args
        hidden_size = args.hidden_size

        self._use_cnn = len(obs_shape) == 3

        input_dim = np.prod(obs_shape)

        print(f"State encoder use_cnn: {self._use_cnn}, input_dim: {input_dim}, output_dim: {output_dim}")

        if self._use_cnn:
            self.cnn = CNNBase(args, obs_shape, mode="encoder")
            input_dim = hidden_size
        else:
            self.mlp = MLPBase(args, input_dim)
            input_dim = hidden_size

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        if gain is None:
            gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.fc = init_(nn.Linear(input_dim, output_dim))
        self.active_func = active_func

    def forward(self, state):
        if self._use_cnn:
            x = self.cnn(state)
        else:
            x = self.mlp(state)
        x = self.fc(x)
        x = self.active_func(x)
        return x


class QmasActor(nn.Module):
    ''' This class modifies the original QMAS actor to include a state encoder. '''
    
    def __init__(self, args, obs_space, share_obs_space, action_space, device=torch.device("cpu")):
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
        self._use_state_encoder = args.state_encoder
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.MAX_NEIGHBORS = 15
        self.MAX_NODES = 50

        self._use_cnn = False
        self._use_mlp = False
        self._use_attention = False
        self._use_rnn = self._use_naive_recurrent_policy or self._use_recurrent_policy
        
        if self._use_gnn:
            # Check whether the env provides dense tensor obs or PyG object obs.
            self._use_gnn_dense_obs = getattr(args, 'gnn_use_dense_obs', False) and not has_graph_obs_space(obs_space)

            if self._use_gnn_dense_obs:
                # Dense obs: the wrapper may have flattened the Dict space into a Box.
                # Recover the original Dict space from the attached attribute if needed.
                dict_space = getattr(obs_space, '_original_dict_space', None) or obs_space
                nf_space = dict_space.spaces["node_features"]   # (max_total_nodes, node_feat_dim)
                ea_space = dict_space.spaces["edge_attr"]        # (max_edges, edge_feat_dim)
                max_total_nodes, node_feat_dim = nf_space.shape
                max_edges, edge_feat_dim = ea_space.shape
                self.gnn_dense_obs_layout = dict(
                    max_total_nodes=max_total_nodes,
                    node_feat_dim=node_feat_dim,
                    edge_feat_dim=edge_feat_dim,
                    max_edges=max_edges,
                )
                # Compute flat obs offsets for each sub-space (gymnasium Dict flatten order).
                self._gnn_dense_offsets = {}
                offset = 0
                for key, space in dict_space.spaces.items():
                    size = int(np.prod(space.shape))
                    self._gnn_dense_offsets[key] = (offset, offset + size, space.shape)
                    offset += size

                # Non-graph obs (observation_radius etc.) — nothing remains after removing graph keys.
                nongraph_dim = 0
                for key, space in dict_space.spaces.items():
                    if key not in ("node_features", "edge_index", "edge_attr", "agent_idx", "neighbors_mask", "num_nodes"):
                        nongraph_dim += int(np.prod(space.shape))

                self.base = GNNBase(
                    layers=args.gnn_layer_N,
                    node_dim=node_feat_dim,
                    edge_dim=edge_feat_dim,
                    hidden_dim=args.gnn_hidden_size,
                    output_dim=args.gnn_hidden_size,
                    dropout_rate=args.gnn_dropout_rate,
                    jk=args.gnn_skip_connections,
                    use_orthogonal=args.use_orthogonal,
                    use_ReLU=args.use_ReLU,
                )

                if args.gnn_neighbor_scoring:
                    # Support the neighbor scoring mechanism from Goeckner et al., DOI: 10.1109/IROS58592.2024.10802510.
                    # When a state encoder is present, the scorer receives per-node GNN features concatenated
                    # with the broadcast global context, so idleness can condition per-neighbor scores.
                    scorer_input_dim = args.gnn_hidden_size
                    if args.state_encoder:
                        scorer_input_dim += args.state_encoder_output_dim
                    self.neighbor_scorer = MLPLayer(input_dim=scorer_input_dim, output_dim=1, hidden_size=self.hidden_size, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False)
                    input_dim = self.MAX_NEIGHBORS + nongraph_dim
                else:
                    input_dim = args.gnn_hidden_size + nongraph_dim

                if self._use_gnn_mlp:
                    if args.state_encoder:
                        input_dim += args.state_encoder_output_dim
                    self.mlp0 = MLPLayer(input_dim=input_dim, output_dim=self.hidden_size, hidden_size=self.hidden_size, layer_N=args.layer_N, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU)
                    input_dim = self.hidden_size
            else:
                # PyG object obs: original path.
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
                    # Support the neighbor scoring mechanism from Goeckner et al., DOI: 10.1109/IROS58592.2024.10802510.
                    # When a state encoder is present, the scorer receives per-node GNN features concatenated
                    # with the broadcast global context, so idleness can condition per-neighbor scores.
                    scorer_input_dim = args.gnn_hidden_size
                    if args.state_encoder:
                        scorer_input_dim += args.state_encoder_output_dim
                    self.neighbor_scorer = MLPLayer(input_dim=scorer_input_dim, output_dim=1, hidden_size=self.hidden_size, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False)
                    input_dim = self.MAX_NEIGHBORS + get_shape_from_obs_space(obs_space_nongraph)[0]
                else:
                    input_dim = args.gnn_hidden_size + get_shape_from_obs_space(obs_space_nongraph)[0]

                if self._use_gnn_mlp:
                    if args.state_encoder:
                        input_dim += args.state_encoder_output_dim
                    self.mlp0 = MLPLayer(input_dim=input_dim, output_dim=self.hidden_size, hidden_size=self.hidden_size, layer_N=args.layer_N, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU)
                    input_dim = self.hidden_size
        else:
            obs_shape = get_shape_from_obs_space(obs_space)
            if args.prediction_uq_injection_method == "append" and not args.state_encoder:
                obs_shape = (*obs_shape[:-1], obs_shape[-1] * 2)

            self._use_cnn = len(obs_shape) == 3
            self._use_mlp = True
            self._use_attention = False

            input_dim = np.prod(obs_shape)

            if self._use_cnn:
                self.cnn = CNNBase(args, obs_shape)
                input_dim = self.hidden_size

            if self._use_attention:            
                self.attention = SelfAttention(input_dim)
                input_dim = input_dim

            if self._use_mlp:
                if args.state_encoder:
                    input_dim += args.state_encoder_output_dim
                self.mlp = MLPBase(args, input_dim)
                input_dim = self.hidden_size
        
        print(f"R_Actor: Use GNN: {self._use_gnn}, Use CNN: {self._use_cnn}, Use MLP: {self._use_mlp}, Use RNN: {self._use_rnn}, Use Encoder: {self._use_state_encoder}")

        # Create the state encoder.
        if args.state_encoder:
            # Determine global observation shape.
            global_obs_shape = get_shape_from_obs_space(share_obs_space)
            if args.prediction_uq_injection_method == "append":
                global_obs_shape = (*global_obs_shape[:-1], global_obs_shape[-1] * 2)

            self.state_encoder = StateEncoder(
                args,
                global_obs_shape,
                output_dim=args.state_encoder_output_dim,
                use_ReLU=args.use_ReLU,
                use_orthogonal=args.use_orthogonal,
                device=device
            )
            # Only add to input_dim when the encoded state is not absorbed by mlp0/mlp.
            if not (self._use_gnn_mlp or self._use_mlp):
                input_dim += args.state_encoder_output_dim

        if self._use_rnn:
            self.rnn = RNNLayer(input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_dim = self.hidden_size

        self.act = ACTLayer(action_space, input_dim, self._use_orthogonal, self._gain)

        self.to(device)


    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, global_obs=None):
        
        actor_features, rnn_states = self._forward(obs, rnn_states, masks, available_actions, global_obs)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states


    def evaluate_actions(self, obs, global_obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action = check(action).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, _ = self._forward(obs, rnn_states, masks, available_actions, global_obs)        

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


    def _forward(self, obs, rnn_states, masks, available_actions=None, global_obs=None):
        '''
        This function performs the bulk of the forward pass to compute the features used for action selection or evaluation.
        It is separated from the main forward function to allow reuse of the feature computation for both action selection and evaluation.
        '''

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # Compute encoded state early so it can be injected into the base models.
        encoded_state = None
        if self._use_state_encoder:
            encoded_state = self._encode_global_state(global_obs, rnn_states.shape[0])

        if self._use_gnn:
            if self._use_gnn_dense_obs:
                # --- Dense tensor obs path ---
                # obs is a float tensor of shape (batch, flat_obs_dim).
                obs_t = check(obs).to(**self.tpdv)
                offsets = self._gnn_dense_offsets
                layout = self.gnn_dense_obs_layout

                def _slice(key):
                    s, e, shape = offsets[key]
                    return obs_t[:, s:e].reshape(-1, *shape)

                node_features_pad = _slice("node_features")  # (B, max_total_nodes, node_feat_dim)
                edge_index_pad    = _slice("edge_index")      # (B, 2, max_edges)
                edge_attr_pad     = _slice("edge_attr")       # (B, max_edges, edge_feat_dim)
                agent_idx_t       = _slice("agent_idx")       # (B, 1)
                nbr_mask_pad      = _slice("neighbors_mask")  # (B, max_total_nodes)
                num_nodes_t       = _slice("num_nodes").squeeze(-1).long()  # (B,)

                # Reconstruct one PyG Data per batch item, excluding padding nodes.
                data_list = []
                for i in range(obs_t.shape[0]):
                    n = int(num_nodes_t[i].item())
                    x_i = node_features_pad[i, :n]            # (n, node_feat_dim)
                    # edge_index: keep only edges where both endpoints < n
                    ei = edge_index_pad[i].long()              # (2, max_edges)
                    ea = edge_attr_pad[i]                      # (max_edges, edge_feat_dim)
                    valid_edges = (ei[0] < n) & (ei[1] < n)
                    ei_i = ei[:, valid_edges]                  # (2, num_real_edges)
                    ea_i = ea[valid_edges]                     # (num_real_edges, edge_feat_dim)
                    nbr_mask_i = nbr_mask_pad[i, :n]          # (n,)

                    d = Data(
                        x=x_i,
                        edge_index=ei_i,
                        edge_attr=ea_i,
                        agent_idx=int(agent_idx_t[i].item()),
                        neighbors_mask=nbr_mask_i.bool().cpu().numpy(),
                    )
                    data_list.append(d)

                graphs = Batch.from_data_list(data_list).to(self.device)
                actor_features = self.base(graphs.x, graphs.edge_attr, graphs.edge_index)
                actor_features, _ = to_dense_batch(actor_features, graphs.batch.to(self.device))

                if self.args.gnn_neighbor_scoring:
                    # Pad to MAX_NODES along the node dimension.
                    if self.MAX_NODES - actor_features.shape[1] > 0:
                        actor_features = F.pad(actor_features, (0, 0, 0, self.MAX_NODES - actor_features.shape[1]), mode='constant', value=0.0)

                    # neighbors_mask from dense obs, padded to MAX_NODES.
                    nbr_mask_full = torch.zeros((obs_t.shape[0], self.MAX_NODES), **self.tpdv)
                    for i in range(obs_t.shape[0]):
                        n = int(num_nodes_t[i].item())
                        nbr_mask_full[i, :n] = nbr_mask_pad[i, :n].float()
                    neighbors_mask_1d = nbr_mask_full.bool()

                    if self._use_state_encoder and encoded_state is not None:
                        encoded_broadcast = encoded_state.unsqueeze(1).expand(-1, actor_features.shape[1], -1)
                        features_for_scorer = torch.cat([actor_features, encoded_broadcast], dim=-1)
                    else:
                        features_for_scorer = actor_features

                    neighbors_mask = neighbors_mask_1d.unsqueeze(2).repeat(1, 1, features_for_scorer.shape[-1])
                    actor_features_masked = torch.where(neighbors_mask, features_for_scorer, 0.0)
                    scores = self.neighbor_scorer(actor_features_masked)

                    scores_shifted = torch.zeros((actor_features.shape[0], self.MAX_NEIGHBORS), **self.tpdv)
                    for i in range(actor_features.shape[0]):
                        nbr_indices = nbr_mask_full[i].nonzero(as_tuple=True)[0][:self.MAX_NEIGHBORS]
                        scores_shifted[i, :nbr_indices.shape[0]] = scores[i, nbr_indices, 0]

                    actor_features = scores_shifted
                else:
                    # Extract agent node features.
                    agent_idx_long = agent_idx_t.long()  # (B, 1)
                    actor_features = self.base.gatherNodeFeats(actor_features, agent_idx_long)

                # Dense obs has no non-graph remainder, but include observation_radius if present.
                s_obs_r, e_obs_r, _ = offsets.get("observation_radius", (0, 0, (0,)))
                if e_obs_r > s_obs_r:
                    obs_radius = obs_t[:, s_obs_r:e_obs_r]
                    actor_features = torch.cat([actor_features, obs_radius], dim=-1)

                obs_nongraph = torch.zeros((obs_t.shape[0], 0), **self.tpdv)

            else:
                # --- PyG object obs path (original) ---
                # Split observation into graph and non-graph components.
                obs_graph = obs[:, self.obs_space_graph_idx]
                nonGraphIdx = [i for i in range(obs.shape[1]) if i != self.obs_space_graph_idx]
                obs_nongraph = obs[:, nonGraphIdx]
                if len(obs_nongraph.shape) > 1 and obs_nongraph.shape[1] > 0:
                    # The non-graph data is stored as an object dtype. Need to convert to float32.
                    obs_non_graph_float = np.zeros((obs_nongraph.shape[0], *obs_nongraph[0, 0].shape), dtype=np.float32)
                    for i in range(obs_nongraph.shape[0]):
                        obs_non_graph_float[i] = obs_nongraph[i, 0]
                    obs_nongraph = obs_non_graph_float
                obs_nongraph = check(obs_nongraph.astype(np.float32)).to(**self.tpdv)

                # Batch the graphs and pass through GNN.
                graphs = Batch.from_data_list(obs_graph).to(self.device, "x", "edge_attr", "edge_index")
                actor_features = self.base(graphs.x, graphs.edge_attr, graphs.edge_index)

                # Restore the original shape of [batch_size, num_nodes (including agents), num_feats] from [batch_size*num_nodes, num_feats]
                actor_features, _ = to_dense_batch(actor_features, graphs.batch.to(self.device))

                # Perform the neighbor scoring from Goeckner et al., DOI: 10.1109/IROS58592.2024.10802510
                if hasattr(graphs, "neighbors") and self.args.gnn_neighbor_scoring:
                    # Pad actor_features to max neighbors.
                    if self.MAX_NODES - actor_features.shape[1] > 0:
                        actor_features = F.pad(actor_features, (0, 0, 0, self.MAX_NODES - actor_features.shape[1]), mode='constant', value=0.0)

                    neighbors_mask_1d = check(np.array(graphs.neighbors_mask)).to(**self.tpdv).bool()  # (batch, MAX_NODES)

                    if self._use_state_encoder and encoded_state is not None:
                        encoded_broadcast = encoded_state.unsqueeze(1).expand(-1, actor_features.shape[1], -1)  # (batch, MAX_NODES, enc_dim)
                        features_for_scorer = torch.cat([actor_features, encoded_broadcast], dim=-1)  # (batch, MAX_NODES, gnn_hidden + enc_dim)
                    else:
                        features_for_scorer = actor_features  # (batch, MAX_NODES, gnn_hidden)

                    # Extend the mask to match the scorer input feature size.
                    neighbors_mask = neighbors_mask_1d.unsqueeze(2).repeat(1, 1, features_for_scorer.shape[-1])
                    actor_features_masked = torch.where(neighbors_mask, features_for_scorer, 0.0)
                    scores = self.neighbor_scorer(actor_features_masked)

                    # Shift the scores to the correct position.
                    scores_shifted = torch.zeros((actor_features.shape[0], self.MAX_NEIGHBORS), **self.tpdv)
                    for i in range(actor_features.shape[0]):
                        nbrs = check(np.array(graphs.neighbors[i])).to(**self.tpdv).int()
                        scores_shifted[i, :nbrs.shape[0]] = scores[i, nbrs, 0]

                    actor_features = scores_shifted

                # Perform evaluation only for a node of interest (typically agent position).
                elif hasattr(graphs, "agent_idx"):
                    agent_idx = torch.from_numpy(np.array(graphs.agent_idx)).reshape(-1, 1).to(self.device)
                    actor_features = self.base.gatherNodeFeats(actor_features, agent_idx)

                # Use the entire graph as the actor features.
                else:
                    actor_features = self.base.graphAggr(actor_features, aggr="mean")

                # Concatenate the graph and non-graph features.
                actor_features = torch.cat([actor_features, obs_nongraph], dim=-1)

            if self._use_gnn_mlp:
                actor_features = torch.cat([actor_features, encoded_state], dim=-1)
                actor_features = self.mlp0(actor_features)
            elif self._use_state_encoder:
                # No mlp0: concat encoded state here before RNN.
                actor_features = torch.cat([actor_features, encoded_state], dim=-1)
        else:
            obs = check(obs).to(**self.tpdv)

            actor_features = obs
            if self._use_cnn:
                actor_features = self.cnn(actor_features)
            if self._use_attention:
                actor_features = self.attention(actor_features)
            if self._use_state_encoder:
                actor_features = torch.cat([actor_features, encoded_state], dim=-1)
            if self._use_mlp:
                actor_features = self.mlp(actor_features)

        # Recurrent network.
        if self._use_rnn:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        return actor_features, rnn_states


    def _encode_global_state(self, global_obs, batch_size):
        '''
        This function encodes the global state using the state encoder.
        batch_size: the per-agent batch size (e.g. n_threads * n_agents).
        '''
        if global_obs is None:
            # If no global state is provided, use zeros.
            encoded_state = torch.zeros((batch_size, self.args.state_encoder_output_dim), device=self.device)
        else:
            global_obs = check(global_obs).to(**self.tpdv)
            global_obs = torch.nan_to_num(global_obs, nan=0.0, posinf=0.0, neginf=0.0)
            # Encode the global state.
            encoded_state = self.state_encoder(global_obs)
            # Repeat the encoded state for each agent if necessary.
            if encoded_state.shape[0] < batch_size:
                if batch_size % encoded_state.shape[0] != 0:
                    raise ValueError("Batch size of obs is not a multiple of batch size of share_obs.")
                encoded_state = encoded_state.repeat_interleave(batch_size // encoded_state.shape[0], dim=0)
        return encoded_state


class QmasCritic(Critic):
    ''' This class is currently the same as the regular MAPPO critic.. '''
    pass