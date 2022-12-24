import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GATv2Conv, LayerNorm, GraphNorm


class GAT(torch.nn.Module):
    """
    Class for the GAT based Graph Stack

        Params:
    layers (int): number of layers in the Stack
    output_dim (int): the size of the output node embeddings
    """
    
    def __init__(self, node_dim: int = 1, edge_dim: int = 1, layers: int = 3, hidden_channels: int = 64, attention_heads: int = 4, attention_dropouts: float = 0.2, dropouts: float = 0.5, graph_norm: bool = True):
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
        
        super(GAT, self).__init__()
        
        self.layers = layers
        
        convs, norms, drops = [], [], []
        for i in range(layers):
            convs.append(GATv2Conv(
                in_channels=hidden_channels if i!=0 else node_dim, 
                out_channels=hidden_channels, 
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
                ))

            if graph_norm == True:
                norms.append(GraphNorm(in_channels=hidden_channels))
            else:
                norms.append(LayerNorm(in_channels=hidden_channels, mode="node"))
                
            drops.append(Dropout(p=dropouts))
            
        self._convs = ModuleList(convs)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)
        
        self.output_dim = hidden_channels
                

    def forward(self, x, edge_index):
        # TODO: return and explore attention
        for i in range(self.layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)
                
        return x

