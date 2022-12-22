from typing import Any, Dict


gat_sweep_configuration = {
    'metric': {
        'goal': 'minimize',
        'name': 'mcrmse'
    },
    'parameters': {
        'subset_size' : {'value': 100} ,
        
        # training params
        'model_name': {'value': 'GAT'},
        'batch_size': {'value': 8},
        'epochs': {'value': 20},
        'lr': {
            'values': [1e-2, 1e-3, 2e-2, 2e-3, 3e-2, 3e-3]
        },
        'loss' : {
            'values': ['crossentropy', 'mse', 'mae']
            },

        # graph params
        'graph_layers': {'values': [3, 5, 9]},
        'graph_hidden_channels': {
            'values': [64, 128, 256, 512]},
        'attention_heads': {
            'values': [2,4,8]
            },
        'attention_dropouts': {
            'values': [0.2, 0.3, 0.5]
            },
        'graph_dropouts': {
            'values': [0.2, 0.3, 0.5]
            },
        'graph_norm': {'value': True},

        # predictor params
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
}

class SweepConfig():
    sweep_configs : Dict[str, Any] = {
        "GAT" : gat_sweep_configuration,
        "GCN" : ...,
        "HeteroGAT": ...,
        "HeteroGCN": ...,
        }
