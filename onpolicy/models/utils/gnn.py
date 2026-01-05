import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import sort_edge_index

from onpolicy.models.utils.gnn_conv import SAGEConvWithEdges

from typing import Tuple, Union, Final
from torch_geometric.nn import AttentionalAggregation, SortAggregation
from onpolicy.models.utils.mlp import MLPLayer

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

        features_channels = node_dim + edge_dim

        # Set up aggregation method.
        if aggr == "attention":
            # Attentional aggregation.
            self.aggr = AttentionalAggregation(
                gate_nn=MLPLayer(
                    input_dim=features_channels,
                    output_dim=1,
                    hidden_size=512,
                    layer_N=3,
                    use_orthogonal=use_orthogonal,
                    use_ReLU=use_ReLU,
                    use_layer_norm=False,
                ),
                nn=MLPLayer(
                    input_dim=features_channels,
                    output_dim=features_channels,
                    hidden_size=256,
                    layer_N=2,
                    use_orthogonal=use_orthogonal,
                    use_ReLU=use_ReLU,
                    use_layer_norm=False,
                ),
            )
        else:
            self.aggr = aggr
        
        # Create the GNN itself.
        self.sage = GraphSAGEWithEdges(
            in_channels=features_channels,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=layers,
            dropout=dropout_rate,
            edge_channels=edge_dim,
            jk="cat" if jk else None,
            aggr=self.aggr,
        )
        if concat_k > 0:
            # Modify the size of the last layer to account for the connection back to hidden size.
            self.sage.convs[-1] = SAGEConvWithEdges(
                in_channels=hidden_dim if layers > 1 else features_channels, # Needs to connect to hidden layer if there is more than 1 layer
                out_channels=output_dim,
                edge_channels=edge_dim,
                concat_k=concat_k,
            )


    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, node_index=None) -> torch.Tensor:

        node_feat = x

        # Add dummy edge attributes for the first layer. In message passing, the edge attributes will be included as part of the node features.
        edge_feat = torch.zeros(node_feat.shape[0], edge_attr.shape[1], device=edge_attr.device)

        # Concatenate node features and type embeddings.
        info = torch.cat([node_feat, edge_feat], dim=1)
        
        # edge_index, edge_attr = sort_edge_index(edge_index, edge_attr, sort_by_row=False)

        return self.sage(info, edge_index, edge_attr=edge_attr)
    

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


class GraphSAGEWithEdges(GraphSAGE):
    r"""All arguments the same as `torch_geometric.nn.GraphSAGE` except for the
    addition of `edge_attr` in the forward method.
    
    The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """

    supports_edge_weight: Final[bool] = False # we just consider weight an edge attribute...
    supports_edge_attr: Final[bool] = True

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        
        return SAGEConvWithEdges(in_channels, out_channels, **kwargs)