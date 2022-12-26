from typing import Dict, Any, List
import torch
from configs.graph_configs import GATConfig, GCNConfig, HANConfig
from model.gat import GAT
from model.gcn import GCN
from model.han import HAN


class PredictorConfig():

    loss_function: Dict[str, Any] = {
        'crossentropy': torch.nn.CrossEntropyLoss,
        'mae': torch.nn.L1Loss,
        'mse': torch.nn.MSELoss
    }

    hyperparameters: Dict[str, Any] = {
        # predictor stack hyperparams
        'pred_layers': {
            'values': [3, 5, 9]
        },
        'pred_hidden_channels': {
            'values': [64, 128, 256, 512]
        },
        'pred_dropouts': {
            'values': [0.2, 0.3, 0.5]
        },
    }


class GraphConfig():
    models: Dict[str, Any] = {
        "GAT": {
            "model": GAT,
            "config": GATConfig,
        },
        "GCN": {
            "model": GCN,
            "config": GCNConfig,
        },
        "HAN": {
            "model": HAN,
            "config": HANConfig,
        },
    }


class TrainConfig():
    hyperparameters = {
        'subset_size': {'value': 1},
        'batch_size': {'values': [8, 16, 32, 64]},
        'epochs': {'value': 20},
        'lr': {
            'values': [1e-2, 1e-3, 2e-2, 2e-3, 3e-2, 3e-3]
        },
        'loss': {
            'values': ['crossentropy', 'mse', 'mae']
        },
    }


# class DataSettings(BaseDataSettings):
#     def __init__(self, data_config: Path) -> None:
#         super(DataSettings).__init__()
#             ...
#                kwargs
#             # TODO load params from json
