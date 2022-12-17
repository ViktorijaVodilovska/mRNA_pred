class HeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_features=1, output_classes=3):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = HeteroConv({
          ('base', 'phosphodiester_bond', 'base'): GCNConv(input_features, hidden_channels),
          ('base', 'hydrogen_bond', 'base'): GCNConv(input_features, hidden_channels),
        }, aggr='sum')
        self.norm1 = LayerNorm(hidden_channels, mode='node')
        self.conv2 = HeteroConv({
          ('base', 'phosphodiester_bond', 'base'): GCNConv(hidden_channels, hidden_channels),
          ('base', 'hydrogen_bond', 'base'): GCNConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        self.norm2 = LayerNorm(hidden_channels, mode='node')
        self.conv3 = HeteroConv({
          ('base', 'phosphodiester_bond', 'base'): GCNConv(hidden_channels, hidden_channels),
          ('base', 'hydrogen_bond', 'base'): GCNConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        self.norm3 = LayerNorm(hidden_channels, mode='node')
        self.lin = Linear(hidden_channels, output_classes)

    def forward(self, x, edge_index):
        # print(x)
        
        x = self.conv1(x, edge_index)
        x['base'] = self.norm1(x['base'])
        # print(x)
        x['base'] = F.dropout(x['base'], p=0.5, training=self.training)
        x['base'] = x['base'].relu()

        x = self.conv2(x, edge_index)
        x['base'] = self.norm2(x['base'])
        # print(x)
        x['base'] = F.dropout(x['base'], p=0.5, training=self.training)
        x['base'] = x['base'].relu()

        x = self.conv3(x, edge_index)
        x['base'] = self.norm3(x['base'])
        # print(x)
        x['base'] = F.dropout(x['base'], p=0.5, training=self.training)
        x['base'] = x['base'].relu()

        # Apply a final classifier
        x = self.lin(x['base'])
        x = torch.sigmoid(x)        
        return x