from typing import Any, Dict
from configs.predictor_config import GraphConfig, PredictorConfig, TrainConfig


class HPOConfig():

    _base_sweep_configuration: Dict[str, Any] = {
        'metric': {
            'goal': 'minimize',
            'name': 'mcrmse'
        },
        'parameters': {
        }
    }

    def __init__(self, model: str = "GAT", algorithm: str = 'random') -> None:

        self.sweep_config = self._base_sweep_configuration
        # TODO: solve for when it isn't random/bayes
        self.sweep_config['method'] = algorithm
        self.sweep_config['parameters'] = GraphConfig.models[model].hyperparameters | PredictorConfig.hyperparameters | TrainConfig.hyperparameters
