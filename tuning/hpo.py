import wandb
from tuning.sweep_configs import SweepConfig
from scripts.experiments import run_training
from config import BaseSettings

# def run_sweep(model_name:str, algorithm: str, project_name: str = "mRNA-graph-models"):
def run_sweep(sweep_id: str = None, model: str = "GAT", algorithm: str = "random", runs: int = 1):
    if sweep_id == None:
        sweep_config = SweepConfig.sweep_configs[model]
        sweep_config['name'] = f"{model}_{algorithm}"
        sweep_config['method'] = algorithm # TODO limit valid options
        sweep_id = wandb.sweep(sweep=sweep_config, project=BaseSettings.PROJECT_NAME)

    print(sweep_id)
    wandb.agent(sweep_id, function=run_experiment, count=runs, project=BaseSettings.PROJECT_NAME)
    

def run_experiment():
    run = wandb.init(project=BaseSettings.PROJECT_NAME, config=wandb.config, resume=True)
    run_training(dict(wandb.config))

if __name__ == "__main__":
    # TODO: ad argspars
    run_sweep(sweep_id='viki/mRNA-graph-models/qgo7p2i7', runs=16)


# start w random/bayesian etc
# shrink space
# go to grid