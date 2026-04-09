import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium.spaces as spaces
import numpy as np

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from onpolicy.models.utils.util import init, check
from onpolicy.models.utils.cnn import CNNBase
from onpolicy.models.utils.mlp import MLPBase, MLPLayer
from onpolicy.models.utils.gnn import GNNBase
from onpolicy.models.utils.rnn import RNNLayer
from onpolicy.models.utils.act import ACTLayer
from onpolicy.models.utils.attention import SelfAttention
from onpolicy.utils.util import get_shape_from_obs_space, get_graph_obs_space, strip_graph_obs_space, get_graph_obs_space_idx
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
        
        if self._use_gnn:
            # Split up the graph and non-graph space.
            obs_space_graph = get_graph_obs_space(obs_space)
            obs_space_nongraph = strip_graph_obs_space(obs_space)
            self.obs_space_graph_idx = get_graph_obs_space_idx(obs_space)

            self.base = GNNBase(
                layers=args.gnn_layer_N,
                node_dim=get_shape_from_obs_space(obs_space_graph.node_space)[0],
                edge_dim=get_shape_from_obs_space(obs_space_graph.edge_space)[0],
                hidden_dim=args.gnn_hidden_size,
                output_dim=args.gnn_hidden_size, #self.hidden_size,
                dropout_rate=args.gnn_dropout_rate,
                jk=args.gnn_skip_connections,
                use_orthogonal=args.use_orthogonal,
                use_ReLU=args.use_ReLU,
            )
            
            if args.gnn_neighbor_scoring:
                # Support the neighbor scoring mechanism from Goeckner et al., DOI: 10.1109/IROS58592.2024.10802510.
                self.neighbor_scorer = MLPLayer(input_dim=args.gnn_hidden_size, output_dim=1, hidden_size=self.hidden_size, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False)
                input_dim = self.MAX_NEIGHBORS + get_shape_from_obs_space(obs_space_nongraph)[0]
            else:
                input_dim = args.gnn_hidden_size + get_shape_from_obs_space(obs_space_nongraph)[0]

            if self._use_gnn_mlp:
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
                self.mlp = MLPBase(args, input_dim)
                input_dim = self.hidden_size
        
        print(f"R_Actor: Use GNN: {self._use_gnn}, Use CNN: {self._use_cnn}, Use MLP: {self._use_mlp}, Use Encoder: {self._use_state_encoder}")

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_dim = self.hidden_size

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
            input_dim += args.state_encoder_output_dim

        self.act = ACTLayer(action_space, input_dim, self._use_orthogonal, self._gain)

        self.to(device)


    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, global_obs=None):
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._use_gnn:
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

                neighbors_mask = check(np.array(graphs.neighbors_mask)).to(**self.tpdv).bool()
                # Extend the mask for the full feature size.
                neighbors_mask = neighbors_mask.unsqueeze(2).repeat(1, 1, actor_features.shape[-1])
                actor_features_masked = torch.where(neighbors_mask, actor_features, 0.0)
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
                actor_features = self.mlp0(actor_features)
        else:
            obs = check(obs).to(**self.tpdv)

            actor_features = obs
            if self._use_cnn:
                actor_features = self.cnn(actor_features)
            if self._use_attention:
                actor_features = self.attention(actor_features)
            if self._use_mlp:
                actor_features = self.mlp(actor_features)

        # Recurrent network.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Handle global observation.
        if self._use_state_encoder:
            if global_obs == None:
                # If no global state is provided, use zeros.
                batch_size = actor_features.shape[0]
                encoded_state = torch.zeros((batch_size, self.args.state_encoder_output_dim), device=actor_features.device)
            else:
                global_obs = check(global_obs).to(**self.tpdv)
                # Encode the global state
                encoded_state = self.state_encoder(global_obs)
                # Repeat the encoded state for each agent if necessary.
                if encoded_state.shape[0] < actor_features.shape[0]:
                    if actor_features.shape[0] % encoded_state.shape[0] != 0:
                        raise ValueError("Batch size of obs is not a multiple of batch size of share_obs.")
                    encoded_state = encoded_state.repeat_interleave(actor_features.shape[0] // encoded_state.shape[0], dim=0)
            actor_features = torch.cat([actor_features, encoded_state], dim=-1)

        # Guard against NaNs in the actor features.
        if torch.any(torch.isnan(actor_features)) or torch.any(torch.isinf(actor_features)):
            print("Warning: NaNs or infinities in actor features during forward pass.")
            actor_features = torch.nan_to_num(actor_features)

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
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if global_obs is not None:
            global_obs = check(global_obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self._use_gnn:
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

                neighbors_mask = check(np.array(graphs.neighbors_mask)).to(**self.tpdv).bool()
                # Extend the mask for the full feature size.
                neighbors_mask = neighbors_mask.unsqueeze(2).repeat(1, 1, actor_features.shape[-1])
                actor_features_masked = torch.where(neighbors_mask, actor_features, 0.0)
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
                actor_features = self.base.graphAggr(actor_features, aggr="max")

            # Concatenate the graph and non-graph features.
            actor_features = torch.cat([actor_features, obs_nongraph], dim=-1)

            if self._use_gnn_mlp:
                actor_features = self.mlp0(actor_features)
        else:
            obs = check(obs).to(**self.tpdv)
            
            actor_features = obs
            if self._use_cnn:
                actor_features = self.cnn(actor_features)
            if self._use_attention:
                actor_features = self.attention(actor_features)
            if self._use_mlp:
                actor_features = self.mlp(actor_features)

        # Recurrent network.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Handle global observation.
        if self._use_state_encoder:
            if global_obs == None:
                # If no global state is provided, use zeros.
                batch_size = actor_features.shape[0]
                encoded_state = torch.zeros((batch_size, self.args.state_encoder_output_dim), device=actor_features.device)
            else:
                # Encode the global state
                encoded_state = self.state_encoder(global_obs)
                # Repeat the encoded state for each agent if necessary.
                if encoded_state.shape[0] < actor_features.shape[0]:
                    if actor_features.shape[0] % encoded_state.shape[0] != 0:
                        raise ValueError("Batch size of obs is not a multiple of batch size of share_obs.")
                    encoded_state = encoded_state.repeat_interleave(actor_features.shape[0] // encoded_state.shape[0], dim=0)
            actor_features = torch.cat([actor_features, encoded_state], dim=-1)

        # Guard against NaNs in the actor features.
        if torch.any(torch.isnan(actor_features)) or torch.any(torch.isinf(actor_features)):
            print("Warning: NaNs or infinities in actor features during evaluation.")
            actor_features = torch.nan_to_num(actor_features)


        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy



class QmasCritic(Critic):
    ''' This class is currently the same as the regular MAPPO critic.. '''
    pass