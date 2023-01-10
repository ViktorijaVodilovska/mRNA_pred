import argparse
import yaml
from pathlib import Path
from typing import Dict
from niapy.problems import Problem
from niapy.task import OptimizationType, Task
from tuning.optimizers import optimizers
from tuning.run import run


# dictionary with the model hpo configurations as yaml files
model_configs: Dict[str, Path] = {
    'GAT' : Path('tuning/hpo_configs/gat_random.yaml'),
    'GCN' : Path('tuning/hpo_configs/gcn_random.yaml'),
    'HAN' : Path('tuning/hpo_configs/han_random.yaml'),
    }


def get_hyperparameters(x, hpo_config):
    """
    Get concrete hyperparameter values for solution `x` from
    the given configuration with hyperparameter options.

    Args:
        x (_type_): _description_
        hpo_config (_type_): _description_

    Returns:
        _type_: _description_
    """

    # for each hyperparam split space into number of options and get option closest to x[i] 

    params = {}
    
    for param_id, (key,vals) in enumerate(hpo_config.items()):
        num_values = len(vals)

        ranges = {
            i:(i*(1/num_values),(i+1)*(1/num_values))
            for i in range(num_values)
        }
        
        for val_id, (start, end) in ranges.items():
            if x[param_id] >= start and x[param_id] < end:
                params[key] = vals[val_id]

    return params


class HyperparameterOptimization(Problem):
    def __init__(self, model_name: str = 'GAT', lower:float = 0, upper: float =1, log=True, group_name:str = 'niapy_hpo'):
        """
        Class defining the hyperparameter optimization problem

        Args:
            model_name (str, optional): _description_. Defaults to 'GAT'.
            lower (float, optional): _description_. Defaults to 0.
            upper (float, optional): _description_. Defaults to 1.
            log (bool, optional): _description_. Defaults to True.
            group_name (str, optional): _description_. Defaults to 'niapy_hpo'.
        """
        
        self.model_name = model_name
        self.group_name = group_name


        with open(model_configs[model_name], "r") as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)['parameters']
            
        self.hpo_config = {k:v['values'] for k,v in parameters.items() if 'values' in v}
        dimension = len(self.hpo_config.keys())

        # TODO: WHAT ARE THESE
        super().__init__(dimension=dimension, lower=lower, upper=upper)

    def _evaluate(self, x):
        run_config = get_hyperparameters(x, self.hpo_config)

        run_config['model_name'] = self.model_name
        run_config['group'] = self.group_name

        model, res = run(**run_config)

        return res['mcrmse'][-1] # TODO: fix hardcoding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing argument")
    parser.add_argument("--algorithm", type=str, default='genetic', help="Optimization algorithm to use. Options [hill, simulated, pso, abc, genetic]")
    parser.add_argument("--model", type=str, default='GAT', help="Model to optimize. Options [GAT, GCN, HAN]")
    args = parser.parse_args()

    problem = HyperparameterOptimization(args.model, group_name=f'{args.model}_{args.algorithm}')

    task = Task(problem, max_iters=1, optimization_type=OptimizationType.MINIMIZATION)

    algorithm = optimizers[args.algorithm]
    best_params, best_score = algorithm.run(task)

    print('Best score:', best_score)
    print('Best parameters:', get_hyperparameters(best_params))
