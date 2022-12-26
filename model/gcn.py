from typing import Any, Dict
import torch
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv
from configs.graph_configs import GCNConfig
from model.graph_stack import GraphStack


class GCN(GraphStack):
    """
    Class for the GCN based Graph Stack
    """

    def __init__(self, node_dim: int = -1, graph_layers: int = 3, graph_hidden_channels: int = 64, graph_dropouts=0.5, graph_norm: bool = True, model_name: str = 'GCN'):
        """
        Creates GCN Stack with specified number of layers, each containing {GCNConv, normalization and dropouts}.

        Args:
            node_dim (int): Size of input node features, or -1 to derive the size from the first input(s) to the forward method. Defaults to -1.
            graph_layers (int, optional): Number of layers in the Stack. Defaults to 3.
            graph_hidden_channels (int, optional): Size of the hidden layers. It also determines output node embedding dimension. Defaults to 64.
            graph_dropouts (float, optional): Rate of dropout after normalization in each layer. Defaults to 0.5.
            graph_norm (bool, optional): Use GraphNorm instead of LayerNorm after each convolution. Defaults to True.
            model_name (str, optional): Name of the built model. Defaults to 'GCN'.
        """
        super(GCN, self).__init__(model_name = model_name, graph_layers = graph_layers,
                                  graph_hidden_channels = graph_hidden_channels, graph_dropouts = graph_dropouts, graph_norm = graph_norm)

        convs = [
            GCNConv(
                in_channels=graph_hidden_channels if layer != 0 else node_dim,
                out_channels=graph_hidden_channels,
                # NOTE: other default params to play with
            )
            for layer in range(graph_layers)
        ]

        self._convs = ModuleList(convs)

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:

        params = {k: config[k] for k in GCNConfig.hyperparameters.keys()}

        params['node_dim'] = graph_info['node_dim']

        return cls(**params)
