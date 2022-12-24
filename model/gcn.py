import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GCNConv, LayerNorm, GraphNorm


class GCN(torch.nn.Module):
    """
    Class for the GCN based Graph Stack

    Params:
    layers (int): number of layers in the Stack
    output_dim (int): the size of the output node embeddings
    """
    
    def __init__(self, node_dim: int = -1, layers: int = 3, hidden_channels: int = 64, dropouts = 0.5, graph_norm: bool = True):
        """
        Creates GCN Stack with specified number of layers, each containing {GCNConv, normalization and dropouts}.

        Args:
            node_dim (int): Size of input node features, or -1 to derive the size from the first input(s) to the forward method. Defaults to -1.
            layers (int, optional): Number of layers in the Stack. Defaults to 3.
            hidden_channels (int, optional): Size of the hidden layers. It also determines output node embedding dimension. Defaults to 64.
            dropouts (float, optional): Rate of dropout after normalization in each layer. Defaults to 0.5.
            graph_norm (bool, optional): Use GraphNorm instead of LayerNorm after each convolution. Defaults to True.
        """
        super(GCN, self).__init__()

        self.layers = layers

        convs, norms, drops = [], [], []
        for i in range(layers):
            convs.append(GCNConv(
                in_channels=hidden_channels if i != 0 else node_dim,
                out_channels=hidden_channels,
                # NOTE: other default params to play with
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
        for i in range(self.layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)

        return x
