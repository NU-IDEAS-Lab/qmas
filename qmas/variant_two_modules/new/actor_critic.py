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
            # Each branch below prepares the flat (compact, no-padding) GNN inputs and
            # any path-specific bookkeeping, then falls through to the shared post-GNN block.
            if self._use_gnn_dense_obs:
                # --- Dense obs: strip padding on CPU, move only compact tensors to GPU ---
                obs_cpu = check(obs)  # numpy → tensor; stays on CPU
                offsets = self._gnn_dense_offsets
                layout  = self.gnn_dense_obs_layout
                B       = obs_cpu.shape[0]
                dev     = self.device

                def _slice_cpu(key):
                    s, e, shape = offsets[key]
                    return obs_cpu[:, s:e].reshape(B, *shape)

                node_features_pad = _slice_cpu("node_features")           # (B, max_N, F)
                edge_index_pad_f  = _slice_cpu("edge_index")              # (B, 2, max_E) float32
                edge_attr_pad     = _slice_cpu("edge_attr")               # (B, max_E, EF)
                agent_idx_t       = _slice_cpu("agent_idx").long()        # (B, 1)
                num_nodes_t       = _slice_cpu("num_nodes").squeeze(-1).long()  # (B,)
                max_N             = layout["max_total_nodes"]

                # Node validity mask and compact node tensor.
                node_range = torch.arange(max_N)
                node_mask  = node_range.unsqueeze(0) < num_nodes_t.unsqueeze(1)  # (B, max_N)
                x_flat_cpu    = node_features_pad[node_mask]                     # (sum_n, F)
                batch_vec_cpu = torch.repeat_interleave(torch.arange(B), num_nodes_t)

                # Cumulative node offsets for global edge index remapping.
                cum_nodes     = torch.zeros(B, dtype=torch.long)
                cum_nodes[1:] = num_nodes_t[:-1].cumsum(0)

                # Edge validity and remapping (float32 arithmetic exact for ints < 2^24).
                n_i_f        = num_nodes_t.float().unsqueeze(1)
                ei_src_f     = edge_index_pad_f[:, 0, :]
                ei_dst_f     = edge_index_pad_f[:, 1, :]
                edge_valid   = (ei_src_f < n_i_f) & (ei_dst_f < n_i_f)
                offset_f     = cum_nodes.float().unsqueeze(1)
                ei_src_g     = (ei_src_f + offset_f)[edge_valid].long()
                ei_dst_g     = (ei_dst_f + offset_f)[edge_valid].long()

                # Move only compact tensors to GPU.
                x_flat          = x_flat_cpu.to(dev)
                edge_index_flat = torch.stack([ei_src_g, ei_dst_g], dim=0).to(dev)
                edge_attr_flat  = edge_attr_pad[edge_valid].to(dev)
                batch_vec       = batch_vec_cpu.to(dev)
                agent_idx_t     = agent_idx_t.to(dev)

                # Neighbor mask (CPU; moved to GPU in shared block if needed).
                nbr_mask_cpu = _slice_cpu("neighbors_mask") if self.args.gnn_neighbor_scoring else None  # (B, max_N)

                # Non-graph remainder (e.g. observation_radius).
                s_r, e_r, _ = offsets.get("observation_radius", (0, 0, (0,)))
                obs_nongraph = obs_cpu[:, s_r:e_r].to(dev) if e_r > s_r else torch.zeros(B, 0, **self.tpdv)

            else:
                # --- PyG object obs path ---
                obs_graph    = obs[:, self.obs_space_graph_idx]
                nonGraphIdx  = [i for i in range(obs.shape[1]) if i != self.obs_space_graph_idx]
                obs_nongraph_np = obs[:, nonGraphIdx]
                if len(obs_nongraph_np.shape) > 1 and obs_nongraph_np.shape[1] > 0:
                    tmp = np.zeros((obs_nongraph_np.shape[0], *obs_nongraph_np[0, 0].shape), dtype=np.float32)
                    for i in range(obs_nongraph_np.shape[0]):
                        tmp[i] = obs_nongraph_np[i, 0]
                    obs_nongraph_np = tmp
                obs_nongraph = check(obs_nongraph_np.astype(np.float32)).to(**self.tpdv)

                graphs = Batch.from_data_list(obs_graph).to(self.device, "x", "edge_attr", "edge_index")
                x_flat          = graphs.x
                edge_index_flat = graphs.edge_index
                edge_attr_flat  = graphs.edge_attr
                batch_vec       = graphs.batch.to(self.device)
                B               = int(batch_vec.max().item()) + 1
                max_N           = self.MAX_NODES

                agent_idx_t  = torch.from_numpy(np.array(graphs.agent_idx)).reshape(-1, 1).to(self.device) \
                               if hasattr(graphs, "agent_idx") else None
                nbr_mask_cpu = check(np.array(graphs.neighbors_mask)) \
                               if (hasattr(graphs, "neighbors") and self.args.gnn_neighbor_scoring) else None

            # ---- Shared post-GNN block ----
            out            = self.base(x_flat, edge_attr_flat, edge_index_flat)  # (sum_n, hidden)
            actor_features, _ = to_dense_batch(out, batch_vec)                  # (B, max_n_i, hidden)
            max_n_i        = actor_features.shape[1]

            if nbr_mask_cpu is not None:
                # Trim / pad neighbor mask to the dense-batch node dimension.
                nbr_cols = nbr_mask_cpu.shape[1]
                if max_n_i <= nbr_cols:
                    nbr_mask_1d = nbr_mask_cpu[:, :max_n_i].bool().to(self.device)
                else:
                    pad = torch.zeros(B, max_n_i - nbr_cols, dtype=torch.bool)
                    nbr_mask_1d = torch.cat([nbr_mask_cpu.bool(), pad], dim=1).to(self.device)

                if self._use_state_encoder and encoded_state is not None:
                    enc_bc = encoded_state.unsqueeze(1).expand(-1, max_n_i, -1)
                    features_for_scorer = torch.cat([actor_features, enc_bc], dim=-1)
                else:
                    features_for_scorer = actor_features

                nbr_mask_3d = nbr_mask_1d.unsqueeze(2).expand_as(features_for_scorer)
                scores = self.neighbor_scorer(
                    torch.where(nbr_mask_3d, features_for_scorer, features_for_scorer.new_zeros(())))

                scores_shifted = torch.zeros(B, self.MAX_NEIGHBORS, **self.tpdv)
                for i in range(B):
                    nbr_idx = nbr_mask_1d[i].nonzero(as_tuple=True)[0][:self.MAX_NEIGHBORS]
                    scores_shifted[i, :nbr_idx.shape[0]] = scores[i, nbr_idx, 0]
                actor_features = scores_shifted

            elif agent_idx_t is not None:
                actor_features = self.base.gatherNodeFeats(actor_features, agent_idx_t)

            else:
                actor_features = self.base.graphAggr(actor_features, aggr="mean")

            actor_features = torch.cat([actor_features, obs_nongraph], dim=-1)

            if self._use_gnn_mlp:
                if self._use_state_encoder and encoded_state is not None:
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