from typing import Any, Dict
import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GCNConv, LayerNorm, GraphNorm
from configs.predictor_config import GraphConfig


class GraphStack(torch.nn.Module):

    def __init__(self, graph_layers: int = 3, graph_hidden_channels: int = 64, graph_dropouts=0.5, graph_norm: bool = True):
        super(GraphStack, self).__init__()

        self.graph_layers = graph_layers

        convs, norms, drops = [], [], []
        for i in range(graph_layers):
            # default model
            convs.append(GCNConv(
                in_channels=-1,
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
        for i in range(self.layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)

        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        model = GraphConfig.models[config['model_name']]['model']
        return model.from_config(config, graph_info)
