import argparse
import yaml
from pathlib import Path
from typing import Dict
import pandas as pd
from scripts.experiments import get_graph_info
from graph.homogenous_dataset import to_pytorch_data
from configs.predictor_config import GraphConfig, PredictorConfig, TrainConfig
from configs.base_config import BaseSettings as settings
from model.predictor import Predictor
from niapy.problems import Problem
from niapy.task import OptimizationType, Task
from scripts.train import train_model
from tuning.optimizers import optimizers
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List


# dictionary with the model hpo configurations as yaml files
model_hpo_configs: Dict[str, Path] = {
    'GAT': Path('tuning/hpo_configs/gat_random.yaml'),
    'GCN': Path('tuning/hpo_configs/gcn_random.yaml'),
    'HAN': Path('tuning/hpo_configs/han_random.yaml'),
}


def get_hyperparameters(x, hpo_config):
    """
    Get concrete hyperparameter values for solution `x` from
    the given configuration with hyperparameter options.
    """

    # for each hyperparam split space into number of options and get option closest to x[i]

    params = {}

    for param_id, (key, vals) in enumerate(hpo_config.items()):
        num_values = len(vals)

        ranges = {
            i: (i*(1/num_values), (i+1)*(1/num_values))
            for i in range(num_values)
        }

        for val_id, (start, end) in ranges.items():
            if x[param_id] >= start and x[param_id] < end:
                params[key] = vals[val_id]

    return params


class HyperparameterOptimization(Problem):
    def __init__(self, train: List[Data], test: List[Data], model_name: str = 'GAT', lower: float = 0, upper: float = 1, log=True, group_name: str = 'niapy_hpo'):
        """
        Class defining the hyperparameter optimization problem
        """

        self.train = train
        self.test = test

        self.model_name = model_name
        self.group_name = group_name

        with open(model_hpo_configs[model_name], "r") as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)['parameters']

        self.hpo_config = {k: v['values']
                           for k, v 
                           in parameters.items() 
                           if 'values' in v
                           }
        dimension = len(self.hpo_config.keys())

        super().__init__(dimension=dimension, lower=lower, upper=upper)

    def _evaluate(self, x):
        # TODO: log
        
        run_config = get_hyperparameters(x, self.hpo_config)

        run_config['model_name'] = self.model_name

        res_conf = {}
        res_conf.update(PredictorConfig.from_dict(run_config))
        res_conf.update(GraphConfig.from_dict(run_config))
        res_conf.update(TrainConfig.from_dict(run_config))

        train_loader = DataLoader(self.train, batch_size=8, shuffle=True)
        val_loader = DataLoader(self.test, batch_size=8, shuffle=True)

        print(res_conf)

        pred_model = Predictor.from_config(res_conf, get_graph_info(self.train[0]))

        res, model = train_model(pred_model,
                                 train_loader,
                                 val_loader,
                                 3,
                                 settings.TARGET_LABELS,
                                 loss_type=res_conf['loss'],
                                 learning_rate=res_conf['lr'],
                                 log=False)

        return res['mcrmse']  # TODO: fix hardcoding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing argument")
    parser.add_argument("--algorithm", type=str, default='genetic', help="Optimization algorithm to use. Options [hill, simulated, pso, abc, genetic]")
    parser.add_argument("--model", type=str, default='GAT', help="Model to optimize. Options [GAT, GCN, HAN]")
    args = parser.parse_args()

    data = pd.read_json(settings.TRAIN_DATA, lines=True).sample(frac=0.01, random_state=0)
    train, val = train_test_split(data, test_size=0.2)

    train_dataset = to_pytorch_data(train, settings.TARGET_LABELS)
    val_dataset = to_pytorch_data(val, settings.TARGET_LABELS)

    problem = HyperparameterOptimization(train_dataset, val_dataset, args.model, group_name=f'{args.model}_{args.algorithm}')

    task = Task(problem, max_iters=20, max_evals = 20, optimization_type=OptimizationType.MINIMIZATION)

    algorithm = optimizers[args.algorithm]
    best_params, best_score = algorithm.run(task)

    print('Best score:', best_score)
    print('Best parameters:', get_hyperparameters(best_params, problem.hpo_config))
