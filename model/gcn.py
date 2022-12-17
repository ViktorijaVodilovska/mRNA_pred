import torch
from torch.nn import Linear, Dropout, LayerNorm
from torch_geometric.nn import GCNConv

class GCNLayer(torch.nn.Module)
    def __init__(self, layers:int, hidden_channels:list[int], input_features:int=1, output_classes:int=3):
            super().__init__()
            
            
            
            self.conv = GCNConv(input_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
            self.conv3 = GCNConv(hidden_channels*2, hidden_channels)

class GCN(torch.nn.Module):
    def __init__(self, layers:int, hidden_channels:list[int], input_features:int=1, output_classes:int=3):
        super().__init__()
        
                
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels)
        self.lin = Linear(hidden_channels, output_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        # Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
            
        return x

model = GCN(hidden_channels=32)
print(model)