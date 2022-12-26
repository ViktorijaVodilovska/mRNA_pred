from typing import Any, Dict
import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GCNConv, LayerNorm, GraphNorm


class GraphStack(torch.nn.Module):
    """
    BaseClass for the Graph Stack.
    Inherits from torch.nn.Module.

    Params:
        graph_layers (int): number of layers in the Stack
        output_dim (int): the size of the output node embeddings
    """

    def __init__(self, model_name: str = 'base', graph_layers: int = 3, graph_hidden_channels: int = 64, graph_dropouts=0.5, graph_norm: bool = True):
        """
        Creates base Graph Stack with specified number of layers, each containing {GCNConv, normalization and dropouts}.

        Args:
            model_name (str, optional): Name of the built model. Defaults to 'base'.
            graph_layers (int, optional): _description_. Defaults to 3.
            graph_hidden_channels (int, optional): _description_. Defaults to 64.
            graph_dropouts (float, optional): _description_. Defaults to 0.5.
            graph_norm (bool, optional): _description_. Defaults to True.
        """
        super(GraphStack, self).__init__()

        self.model_name = model_name
        self.graph_layers = graph_layers

        convs, norms, drops = [], [], []
        for i in range(graph_layers):
            # default model
            convs.append(GCNConv(
                in_channels=-1 if i == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
            ))

            if graph_norm == True:
                norms.append(GraphNorm(in_channels=graph_hidden_channels))
            else:
                norms.append(
                    LayerNorm(in_channels=graph_hidden_channels, mode="node"))

            drops.append(Dropout(p=graph_dropouts))

        self._convs = ModuleList(convs)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)

        self.output_dim = graph_hidden_channels

    def forward(self, x, edge_index):
        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)

        return x
