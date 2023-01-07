from typing import Dict, Any, cast
import torch
from configs.graph_configs import GATConfig, GCNConfig, HANConfig
from model.gat import GAT
from model.gcn import GCN
from model.han import HAN


class PredictorConfig:

    hyperparameters: Dict[str, Any] = {
        # predictor stack hyperparams
        "pred_layers": 3,
        "pred_hidden_channels": 64,
        "pred_dropouts": 0.3,
    }

    @classmethod
    def from_dict(cls, config: Dict['str', Any]):

        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val

        return res_conf


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

    @classmethod
    def from_dict(cls, config: Dict['str', Any]):

        return cls.models[config['model_name']]['config'].from_dict(config)


class TrainConfig():

    loss_function: Dict[str, Any] = {
        'crossentropy': torch.nn.CrossEntropyLoss,
        'mae': torch.nn.L1Loss,
        'mse': torch.nn.MSELoss
    }

    hyperparameters = {
        "subset_size": 0.1,
        "batch_size": 64,
        "epochs": 1,
        "lr": 1e-3,
        "loss": "crossentropy",
    }

    @classmethod
    def from_dict(cls, config: Dict['str', Any]):

        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val

        return res_conf
