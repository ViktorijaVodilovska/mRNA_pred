import torch
from torch.nn import Dropout, Linear, ModuleList
from torch_geometric.nn import LayerNorm
from typing import Any, Dict
from configs.base_config import BaseSettings as settings
from configs.predictor_config import GraphConfig
from model.graph_stack import GraphStack


class Predictor(torch.nn.Module):
    """
    Class for the Graph Predictor containing a graph model and a prediction head.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, graph_model: torch.nn.Module, pred_layers: int = 3, pred_hidden_channels: int = 64, output_channels: int = 5, pred_dropouts: float = None):
        """
        Creates a Graph Prediction Stack with the specified graph stack and a prediction stack with the given number of layers, each containing {feed forward layer, normalization and dropouts}.

        Args:
            graph_model (torch.nn.Module): _description_
            pred_layers (int, optional): Number of layers in the prediction Stack. Defaults to 3.
            pred_hidden_channels (int, optional): Size of the hidden layers. Defaults to 64.
            output_channels (int, optional): Number of values to predict. Defaults to 5.
            pred_dropouts (float, optional): Rate of dropout after normalization in each layer. Defaults to 0.5.
        """
        super(Predictor, self).__init__()

        self.graph_stack = graph_model
        self.pred_layers = pred_layers

        self._nns = ModuleList(
            [Linear(
                in_features=pred_hidden_channels if i != 0 else graph_model.output_dim,
                out_features=pred_hidden_channels if i != pred_layers-1 else output_channels)
             for i in range(pred_layers)]
        )

        self._norms = ModuleList(
            [LayerNorm(pred_hidden_channels) for i in range(pred_layers-1)]
        )

        self._drops = ModuleList(
            [Dropout(pred_dropouts) for i in range(pred_layers-1)]
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:

        graph_model = GraphConfig.models[config['model_name']]['model'].from_config(config, graph_info)

        pred_model = Predictor(graph_model=graph_model,
                               pred_layers=config['pred_layers'],
                               pred_hidden_channels=config['pred_hidden_channels'],
                               output_channels=len(settings.TARGET_LABELS),
                               pred_dropouts=config['pred_dropouts'])

        return pred_model

    def forward(self, data):

        x = self.graph_stack(data.x, data.edge_index)

        for i in range(self.pred_layers-1):
            x = self._nns[i](x)
            x = self._norms[i](x)
            x = self._drops[i](x)

        x = self._nns[-1](x)

        return x
