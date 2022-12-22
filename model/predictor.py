import argparse
import torch
from torch.nn import Dropout, Linear, ModuleList
from torch_geometric.nn import GATv2Conv, LayerNorm, GraphNorm
from typing import Union, Dict
import wandb


class Predictor(torch.nn.Module):
    def __init__(self, graph_model, layers, hidden_channels, output_channels, dropouts=None):
        super(Predictor, self).__init__()
        
        self.graph_stack = graph_model
        self.layers = layers
   
        self._nns = ModuleList(
            [Linear(
                in_features=hidden_channels if i!=0 else graph_model.output_dim,
                out_features=hidden_channels if i!=layers-1 else output_channels)
            for i in range(layers)]
        )
        
        self._norms = ModuleList(
            [LayerNorm(hidden_channels) for i in range(layers-1)]
        )
        
        self._drops = ModuleList(
            [Dropout(dropouts) for i in range(layers-1)]
        )
        
        
    def forward(self, data):
        
        x = self.graph_stack(data.x, data.edge_index)
        
        for i in range(self.layers-1):
            x = self._nns[i](x)
            x = self._norms[i](x)
            x = self._drops[i](x)
        
        x = self._nns[-1](x)
        
        return x  