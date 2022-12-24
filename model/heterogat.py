import torch
from typing import Union, Dict, Tuple, List
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import HANConv, LayerNorm, GraphNorm


class HeteroGAT(torch.nn.Module):
    """
    Class for the HAN based Graph Stack

    Params:
    layers (int): number of layers in the Stack
    output_dim (int): the size of the output node embeddings
    """
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], in_channels: Union[int, Dict[str, int]]=-1, layers: int = 3, hidden_channels: int = 64, attention_heads: int = 4, attention_dropouts: float = 0.2, dropouts: float = 0.5, graph_norm: bool = True):                 
        """
        Creates HAN Stack with specified number of layers, each containing {HANConv, normalization and dropouts}.

        Args:
            metadata (Tuple): The metadata of the heterogeneous graph, i.e. its node and edge types. See torch_geometric.data.HeteroData.metadata().
            in_channels (Union[int, Dict[str, int]], optional): Size of each input sample of every node type, or -1 to derive the size from the first input(s) to the forward method.
            layers (int, optional): Number of layers in the Stack. Defaults to 3.
            hidden_channels (int, optional): Size of the hidden layers. It also determines output node embedding dimension. Defaults to 64.
            attention_heads (int, optional): Number of attention heads for all the layers. Defaults to 4.
            attention_dropouts (float): Rate of dropout of the attention coeficients in each layer. Defaults to 0.5.
            dropouts (float, optional): Rate of dropout after normalization in each layer. Defaults to 0.5.
            graph_norm (bool, optional): Use GraphNorm instead of LayerNorm after each convolution. Defaults to True.
        """
        super(HeteroGAT, self).__init__()
        
        self.layers = layers

        convs, norms, drops = [], [], []
        for i in range(layers):
            convs.append(HANConv(
                in_channels = in_channels if i==0 else hidden_channels, 
                out_channels = hidden_channels, 
                heads=attention_heads, 
                dropout=attention_dropouts, 
                metadata=metadata,
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
        # TODO: return and explore attention
        for i in range(self.layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)
                
        return x

