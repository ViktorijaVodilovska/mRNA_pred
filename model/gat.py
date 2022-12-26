from typing import Any, Dict
import torch
from torch.nn import ModuleList
from torch_geometric.nn import GATv2Conv
from configs.graph_configs import GATConfig
from model.graph_stack import GraphStack


class GAT(GraphStack):
    """
    Class for the GAT based Graph Stack

        Params:
    layers (int): number of layers in the Stack
    output_dim (int): the size of the output node embeddings
    """
    
    def __init__(self, node_dim: int = 1, edge_dim: int = 1, graph_layers: int = 3, graph_hidden_channels: int = 64, attention_heads: int = 4, attention_dropouts: float = 0.2, graph_dropouts: float = 0.5, graph_norm: bool = True):
        """
        Creates GAT Stack with specified number of layers, each containing {GATConv, normalization and dropouts}.

        Args:
            node_dim (int, optional): Size of input node features. Defaults to 1.
            edge_dim (int, optional): Size of input node features. Defaults to 1.
            layers (int, optional): Number of layers in the Stack. Defaults to 3.
            hidden_channels (int, optional): Size of the hidden layers. It also determines output node embedding dimension. Defaults to 64.
            attention_heads (int, optional): Number of attention heads for all the layers. Defaults to 4.
            attention_dropouts (float): Rate of dropout of the attention coeficients in each layer. Defaults to 0.2.
            dropouts (float, optional): Rate of dropout after normalization in each layer. Defaults to 0.5.
            graph_norm (bool, optional): Use GraphNorm instead of LayerNorm after each convolution. Defaults to True.
        """
        super(GAT, self).__init__(graph_layers, graph_hidden_channels, graph_dropouts, graph_norm)

        convs = [
            GATv2Conv(
                in_channels=graph_hidden_channels if layer!=0 else node_dim, 
                out_channels=graph_hidden_channels, 
                heads=attention_heads, 
                dropout=attention_dropouts,
                edge_dim=edge_dim,
                # TODO: alternatives to try for concat: concat=True / is concatenated and projected (NN) back down to a size of 64 
                concat = False,

                # NOTE: default params to play with:
                # return_attention_weights=True if i==layers-1 else False,
                # negative_slope 
                # add_self_loops 
                # fill_value 
                # bias 
                # share_weights 
                )
            for layer in range(graph_layers)
            ]

        self._convs = ModuleList(convs)

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info : Dict[str, Any]) -> torch.nn.Module:

        params = {k:config[k] for k in GATConfig.hyperparameters.keys()}

        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info['edge_dim']

        return cls(**params)