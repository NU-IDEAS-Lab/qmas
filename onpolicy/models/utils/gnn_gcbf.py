import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from onpolicy.models.utils.mlp import MLPLayer
from onpolicy.models.utils.util import init as util_init

"""GNN modules."""

class GNNBase(nn.Module):
    ''' Base GNN module. '''

    def __init__(self, layers: int, node_dim: int, edge_dim: int, output_dim: int, hidden_dim: int,
                 dropout_rate: float = 0.0,
                 jk = False,
                 aggr = "attention",
                 concat_k = 0,
                 use_orthogonal = True,
                 use_ReLU = True,
                 **kwargs):
        super(GNNBase, self).__init__(**kwargs)

        self.gnn = GCBFMessagePassing(
            node_dim=node_dim,
            edge_dim=edge_dim,
            msg_hidden_dim=hidden_dim,
            msg_out_dim=hidden_dim,
            attn_hidden_dim=hidden_dim,
            node_hidden_dim=hidden_dim,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        def _init_(m):
            return util_init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.output_layer = _init_(nn.Linear(node_dim, output_dim))


    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, node_index=None) -> torch.Tensor:
        out = self.gnn(x, edge_index, edge_attr=edge_attr)
        out = self.output_layer(out)
        return out
    

    def gatherNodeFeats(self, x: torch.Tensor, idx: torch.Tensor):
        """
        This method is borrowed from InforMARL: https://github.com/nsidn98/InforMARL/blob/main/onpolicy/algorithms/utils/gnn.py#L346

        The output obtained from the network is of shape
        [batch_size, num_nodes, out_channels]. If we want to
        pull the features according to particular nodes in the
        graph as determined by the `idx`, use this
        Refer below link for more info on `gather()` method for 3D tensors
        https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Args:
            x (Tensor): Tensor of shape (batch_size, num_nodes, out_channels)
            idx (Tensor): Tensor of shape (batch_size) or (batch_size, k)
                indicating the indices of nodes to pull from the graph

        Returns:
            Tensor: Tensor of shape (batch_size, out_channels) which just
                contains the features from the node of interest
        """
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)  # (batch_size, 1)
            assert idx_tmp.shape == (batch_size, 1)
            idx_tmp = idx_tmp.repeat(1, num_feats)  # (batch_size, out_channels)
            idx_tmp = idx_tmp.unsqueeze(1)  # (batch_size, 1, out_channels)
            gathered_node = x.gather(1, idx_tmp).squeeze(1)  # (batch_size, out_channels)
            out.append(gathered_node)
        out = torch.cat(out, dim=1)  # (batch_size, out_channels*k)

        return out


    def graphAggr(self, x: torch.Tensor, aggr: str = "mean"):
        """
        This method is borrowed from InforMARL: https://github.com/nsidn98/InforMARL/blob/main/onpolicy/algorithms/utils/gnn.py#L381

        Aggregate the graph node features by performing global pool


        Args:
            x (Tensor): Tensor of shape [batch_size, num_nodes, num_feats]
            aggr (str): Aggregation method for performing the global pool

        Raises:
            ValueError: If `aggr` is not in ['mean', 'max']

        Returns:
            Tensor: The global aggregated tensor of shape [batch_size, num_feats]
        """
        if aggr == "mean":
            return x.mean(dim=1)
        elif aggr == "max":
            max_feats, idx = x.max(dim=1)
            return max_feats
        elif aggr == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"`aggr` should be one of 'mean', 'max', 'add'")


class GCBFMessagePassing(MessagePassing):
    """
    PyTorch Geometric rewrite of gcbfplus/nn/gnn.py

    Matches:
    - message MLP: concat(sender, receiver, edge)
    - attention MLP -> scalar weights
    - softmax aggregation over incoming edges
    - node update MLP
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        msg_hidden_dim: int,
        msg_out_dim: int,
        attn_hidden_dim: int,
        node_hidden_dim: int,
        use_orthogonal: bool = True,
        use_ReLU: bool = True,
    ):
        super().__init__(aggr="add")  # aggregation handled manually

        self.message_mlp = MLPLayer(
            input_dim=2 * node_dim + edge_dim,
            output_dim=msg_out_dim,
            hidden_size=msg_hidden_dim,
            layer_N=2,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

        self.attn_mlp = MLPLayer(
            input_dim=msg_out_dim,
            output_dim=1,
            hidden_size=attn_hidden_dim,
            layer_N=2,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

        self.node_mlp = MLPLayer(
            input_dim=node_dim + msg_out_dim,
            output_dim=node_dim,
            hidden_size=node_hidden_dim,
            layer_N=2,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, node_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):
        """
        x_i: receiver node features [E, node_dim]
        x_j: sender node features   [E, node_dim]
        index: target node indices for each edge [E]
        """
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m = self.message_mlp(msg_input)  # [E, msg_out_dim]

        attn_logits = self.attn_mlp(m).squeeze(-1)  # [E]
        attn = softmax(attn_logits, index=index)

        return m * attn.unsqueeze(-1)

    def update(self, aggr_out, x):
        """
        aggr_out: aggregated messages [N, msg_out_dim]
        x: original node features [N, node_dim]
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)
