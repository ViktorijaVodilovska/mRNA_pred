import argparse
import yaml
import gc
import wandb
import pandas as pd
import multiprocessing as mp
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from configs.base_config import BaseSettings as settings
from graph.mrna_dataset import mRNADataset
from tuning.optimizers import model_hpo_configs
from model.predictor import Predictor
from scripts.experiments import get_graph_info
from scripts.train import train_model

# centralized data
# TODO: make singleton
data = pd.read_json(settings.TRAIN_DATA, lines=True).sample(
        frac=0.1, random_state=0)
train, val = train_test_split(data, test_size=0.2)
train=train.reset_index()
val=val.reset_index()
    
train_dataset = mRNADataset(data = train, target_cols = settings.TARGET_LABELS)
val_dataset = mRNADataset(data = val, target_cols = settings.TARGET_LABELS)

# Define (objective) training function
def train(config):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    # make model
    print("First entry example:")
    print(train_dataset[0])
    graph_info = get_graph_info(train_dataset[0], False)
    pred_model = Predictor.from_config(config, graph_info)

    # train model
    res, model = train_model(pred_model,
                             train_loader,
                             val_loader,
                             config['epochs'],
                             settings.TARGET_LABELS,
                             loss_type=config['loss'],
                             learning_rate=config['lr'],
                             hetero=False,
                             log=True,
                             save_to=None)

    return res, model

def main():
    print('in main!')
    wandb.init()
    config = wandb.config
    print(config)
    res,model = train(config)
    wandb.finish()
    gc.collect

def run_agent(sweep_id: str):
    print('in agent!')
    wandb.agent(sweep_id = sweep_id, function=main)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parsing argument")
    parser.add_argument("--method", type=str, default='random', help="Optimization method to use. Options [random, bayes, grid]")
    parser.add_argument("--model", type=str, default='GAT', help="Model to optimize. Options [GAT, GCN]")
    parser.add_argument("--count", type=int, default=None, help="Number of generated solutions")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for optimization")
    parser.add_argument("--experiment_name", type=str, default="test_sweep", help="Name to add to logged experiment.")
    args = parser.parse_args()

    with open(model_hpo_configs[args.model], "r") as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)['parameters']

    sweep_configuration = {
        "name": f"{args.experiment_name}_{args.model}_{args.method}",
        "metric": {"name": "mcmrse", "goal": "minimize"},
        "method": args.method,
        "parameters": parameters
        }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=settings.PROJECT_NAME)
    if args.count == None:
        # Create a child process to run the algorithm
        p = mp.Process(target=run_agent, args=({sweep_id}))
        p.start()
        
        # Wait for the process to finish or timeout
        p.join(timeout=args.timeout)
        print('now back here!')

        # If the process is still alive after the timeout, terminate it
        if p.is_alive():
            p.terminate()
    else:
        wandb.agent(sweep_id = sweep_id, function=main, count=args.count)