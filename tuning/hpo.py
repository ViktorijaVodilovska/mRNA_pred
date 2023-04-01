import argparse
import yaml
from pathlib import Path
from typing import Dict
import pandas as pd
import wandb
from graph.mrna_dataset import mRNADataset
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
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import List
import gc
import multiprocessing as mp


# dictionary with the model hpo configurations as yaml files
model_hpo_configs: Dict[str, Path] = {
    'GAT': Path('tuning/hpo_configs/gat_10_narrow.yaml'),
    'GCN': Path('tuning/hpo_configs/gcn_10_narrow.yaml'),
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

        if num_values == 1:
            params[key] = vals

        ranges = {
            i: (i*(1/num_values), (i+1)*(1/num_values))
            for i in range(num_values)
        }

        for val_id, (start, end) in ranges.items():
            if x[param_id] >= start and x[param_id] < end:
                params[key] = vals[val_id]

    return params

def evaluate_model(train, test, config, log):
    train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(test, batch_size=config['batch_size'], shuffle=True)

    pred_model = Predictor.from_config(config, get_graph_info(train[0]))

    res, model = train_model(pred_model,
                                 train_loader,
                                 val_loader,
                                 epochs=config['epochs'],
                                 target_labels = settings.TARGET_LABELS,
                                 loss_type=config['loss'],
                                 learning_rate=config['lr'],
                                 log=log)

    return res['mcrmse']
    


class HyperparameterOptimization(Problem):
    def __init__(self, train: Dataset, test: Dataset, model_name: str = 'GAT', lower: float = 0, upper: float = 1, log:bool=False, group_name: str = 'niapy_hpo'):
        """
        Class defining the hyperparameter optimization problem
        """

        self.train = train
        self.test = test

        self.model_name = model_name
        self.group_name = group_name

        self.log = log

        with open(model_hpo_configs[model_name], "r") as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)['parameters']

        self.hpo_config = {k: v['values']
                           for k, v 
                           in parameters.items() 
                           if 'values' in v
                           }
        self.exp_config = {k: v['value']
                           for k, v 
                           in parameters.items() 
                           if 'value' in v
                           }
        dimension = len(self.hpo_config.keys())

        super().__init__(dimension=dimension, lower=lower, upper=upper)

    def _evaluate(self, x):
        
        run_config = get_hyperparameters(x, self.hpo_config)
        run_config.update(self.exp_config)

        res_conf={}

        res_conf.update(PredictorConfig.from_dict(run_config))
        res_conf.update(GraphConfig.from_dict(run_config))
        res_conf.update(TrainConfig.from_dict(run_config))

        print(res_conf)

        if self.log:
            run = wandb.init(config=res_conf, group=self.group_name, project=settings.PROJECT_NAME)

        res = evaluate_model(self.train, self.test, res_conf, self.log)
        
        if self.log:
            wandb.finish()

        gc.collect

        return res  # TODO: fix hardcoding


def run_algorithm(task, algorithm):
    best_params, best_score = algorithm.run(task)
    return best_params, best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing argument")
    parser.add_argument("--algorithm", type=str, default='genetic', help="Optimization algorithm to use. Options [hill, simulated, pso, abc, genetic]")
    parser.add_argument("--model", type=str, default='GAT', help="Model to optimize. Options [GAT, GCN, HAN]")
    parser.add_argument("--evals", type=int, default=None, help="Number of model evaluations")
    parser.add_argument("--iters", type=int, default=None, help="Number of generated solutions")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for optimization")
    parser.add_argument("--log", type=bool, default=False, help="Log to wandb?")
    parser.add_argument("--experiment_name", type=str, default="HPO", help="Name to add to logged experiment.")
    parser.add_argument("--data_fraction", type=float, default=0.1, help="Fraction of data to include for experiment.")
    
    args = parser.parse_args()

    data = pd.read_json(settings.TRAIN_DATA, lines=True).sample(frac=args.data_fraction, random_state=0)
    train, val = train_test_split(data, test_size=0.2)

    train=train.reset_index()
    val=val.reset_index()

    train_dataset = mRNADataset(data = train, target_cols = settings.TARGET_LABELS)
    val_dataset = mRNADataset(data = val, target_cols = settings.TARGET_LABELS)

    problem = HyperparameterOptimization(train_dataset, val_dataset, args.model, group_name=f'{args.experiment_name}_{args.model}_{args.algorithm}', log=args.log)    
    algorithm = optimizers[args.algorithm]

    if args.timeout == None:
        # Run for fixed iterations
        task = Task(problem, max_iters=args.iters, max_evals = args.evals, optimization_type=OptimizationType.MINIMIZATION)
        best_params, best_score = algorithm.run(task)
    else:
        # Run for fixed time, unlimited iterations
        task = Task(problem, optimization_type=OptimizationType.MINIMIZATION)

        # Create a child process to run the algorithm
        p = mp.Process(target=run_algorithm, args=(task, algorithm))
        p.start()

        print('here!')
        
        # Wait for the process to finish or timeout
        p.join(timeout=args.timeout)

        print('now here!')

        if p.is_alive():
            # If the process is still alive after the timeout, terminate it
            p.terminate()
