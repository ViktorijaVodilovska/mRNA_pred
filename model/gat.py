import torch
from typing import Union, Dict
from torch_geometric.nn import Linear
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import LayerNorm

class GAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, attention_heads, dropout, output_classes):
        super(GAT, self).__init__()

        # conv + attention emb + linear
        self.conv1 = GATv2Conv(
            in_channels=node_dim, 
            out_channels=hidden_channels, 
            heads=attention_heads, 
            dropout=dropout,
            edge_dim=edge_dim)
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels*attention_heads,
            out_channels=hidden_channels, 
            concat=False,
            heads=attention_heads, 
            dropout=0)
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels*attention_heads,
            out_channels=hidden_channels, 
            concat=False,
            heads=attention_heads, 
            dropout=0)
        self.lin = Linear(hidden_channels, output_classes)
        

    def forward(self,x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        # Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # x = torch.sigmoid(x)        
        return x