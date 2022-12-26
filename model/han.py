import torch
from typing import Union, Dict, Tuple, Any, List
from torch.nn import ModuleList
from torch_geometric.nn import HANConv
from configs.graph_configs import HANConfig
from model.graph_stack import GraphStack


class HAN(GraphStack):
    """
    Class for the HAN based Graph Stack
    """

    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], in_channels: Union[int, Dict[str, int]] = -1, graph_layers: int = 3, graph_hidden_channels: int = 64, attention_heads: int = 4, attention_dropouts: float = 0.2, graph_dropouts: float = 0.5, graph_norm: bool = True,  model_name: str = 'HAN'):
        """
        Creates HAN Stack with specified number of layers, each containing {HANConv, normalization and dropouts}.

        Args:
            metadata (Tuple): The metadata of the heterogeneous graph, i.e. its node and edge types. See torch_geometric.data.HeteroData.metadata().
            in_channels (Union[int, Dict[str, int]], optional): Size of each input sample of every node type, or -1 to derive the size from the first input(s) to the forward method.
            graph_layers (int, optional): Number of layers in the Stack. Defaults to 3.
            graph_hidden_channels (int, optional): Size of the hidden layers. It also determines output node embedding dimension. Defaults to 64.
            attention_heads (int, optional): Number of attention heads for all the layers. Defaults to 4.
            attention_dropouts (float): Rate of dropout of the attention coeficients in each layer. Defaults to 0.5.
            graph_dropouts (float, optional): Rate of dropout after normalization in each layer. Defaults to 0.5.
            graph_norm (bool, optional): Use GraphNorm instead of LayerNorm after each convolution. Defaults to True.
            model_name (str, optional): Name of the built model. Defaults to 'HAN'.
            
        """
        super(HAN, self).__init__(model_name = model_name, graph_layers = graph_layers,
                                  graph_hidden_channels = graph_hidden_channels, graph_dropouts = graph_dropouts, graph_norm = graph_norm)

        convs = [
            HANConv(
                in_channels=in_channels if layer == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
                heads=attention_heads,
                dropout=attention_dropouts,
                metadata=metadata,
                # NOTE: other default params to play with
            )
            for layer in range(graph_layers)
        ]

        self._convs = ModuleList(convs)

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:

        params = {k: config[k] for k in HANConfig.hyperparameters.keys()}

        params['in_channels'] = graph_info['in_channels']
        params['metadata'] = graph_info['metadata']

        return cls(**params)
