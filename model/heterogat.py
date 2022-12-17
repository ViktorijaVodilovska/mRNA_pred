import torch
from typing import Union, Dict
from torch_geometric.nn import Linear
from torch_geometric.nn import HANConv
from torch_geometric.nn import LayerNorm

class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata: tuple,in_channels: Union[int, Dict[str, int]]=-1, out_channels: int=3, hidden_channels:int=64, heads:int =8, dropout: float=0.5):                 
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = HANConv(in_channels=in_channels, out_channels = hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.norm1 = LayerNorm(hidden_channels, mode='node')

        self.conv2 = HANConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.norm2 = LayerNorm(hidden_channels, mode='node')

        self.conv3 = HANConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.norm3 = LayerNorm(hidden_channels, mode='node')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x['base'] = self.norm1(x['base'])
        # x['base'] = F.dropout(x['base'], p=0.5, training=self.training)
        # x['base'] = x['base'].relu()

        x = self.conv2(x, edge_index)
        x['base'] = self.norm2(x['base'])
        # x['base'] = F.dropout(x['base'], p=0.5, training=self.training)
        # x['base'] = x['base'].relu()

        x = self.conv3(x, edge_index)
        x['base'] = self.norm3(x['base'])
        # x['base'] = F.dropout(x['base'], p=0.5, training=self.training)
        # x['base'] = x['base'].relu()

        # Apply a final classifier
        x = self.lin(x['base'])
        
        # ???????????
        x = torch.softmax(x)    # NE SOFTMAX    
        return x