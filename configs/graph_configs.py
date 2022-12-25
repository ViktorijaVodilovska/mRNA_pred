from typing import Any, Dict


class GATConfig():
    hyperparameters = {
        'model_name': {'value': 'GAT'},
        'graph_layers': {'values': [3, 5, 9]},
        'graph_hidden_channels': {
            'values': [64, 128, 256, 512]},
        'attention_heads': {
            'values': [2, 4, 8]
        },
        'attention_dropouts': {
            'values': [0.2, 0.3, 0.5]
        },
        'graph_dropouts': {
            'values': [0.2, 0.3, 0.5]
        },
        'graph_norm': {'value': True},
    }

class GCNConfig():
    hyperparameters = {
        'model_name': {'value': 'GCN'},
        'graph_layers': {'values': [3, 5, 9]},
        'graph_hidden_channels': {
            'values': [64, 128, 256, 512]},
        'graph_dropouts': {
            'values': [0.2, 0.3, 0.5]
        },
        'graph_norm': {'value': True},
    }

class HANConfig():
    hyperparameters = {
        'model_name': {'value': 'HAN'},
        'graph_layers': {'values': [3, 5, 9]},
        'graph_hidden_channels': {
            'values': [64, 128, 256, 512]},
        'attention_heads': {
            'values': [2, 4, 8]
        },
        'attention_dropouts': {
            'values': [0.2, 0.3, 0.5]
        },
        'graph_dropouts': {
            'values': [0.2, 0.3, 0.5]
        },
        'graph_norm': {'value': True},
    }