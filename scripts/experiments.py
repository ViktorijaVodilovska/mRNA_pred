from model.gat import GAT
from model.predictor import Predictor
from torch_geometric.loader import DataLoader
from graph.homogenous_dataset import to_pytorch_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Any
from config import BaseSettings as settings
from scripts.train import train_model
import wandb

def run_training(config: Dict[str, Any], log: bool =True):

    print(config)

    # Load train and val data
    data = pd.read_json(settings.TRAIN_DATA, lines=True)[:config['subset_size']]
    train, val = train_test_split(data, test_size=0.2)

    train_dataset = to_pytorch_dataset(train, settings.BPPM_FOLDER, settings.TARGET_LABELS)
    val_dataset = to_pytorch_dataset(val, settings.BPPM_FOLDER, settings.TARGET_LABELS)
    print("First entry example:")
    print(train_dataset[0])

    node_dim = train_dataset[0].x.shape[1]
    edge_dim = train_dataset[0].edge_attr.shape[0]

    # make a data loader for batches
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    # make model
    # get model from name    
    graph_model = GAT(node_dim=node_dim,
                      edge_dim=edge_dim,
                      layers=config['graph_layers'],
                      hidden_channels=config['graph_hidden_channels'],
                      attention_heads=config['attention_heads'],
                      attention_dropouts=config['attention_dropouts'],
                      dropouts=config['attention_dropouts'],
                      graph_norm=config['graph_norm'])

    pred_model = Predictor(graph_model=graph_model,
                           layers=config['pred_layers'],
                           hidden_channels=config['pred_hidden_channels'],
                           output_channels=5,
                           dropouts=config['pred_dropouts'])

    train_model(pred_model, 
                train_loader, 
                val_loader,
                config['epochs'], 
                settings.TARGET_LABELS, 
                loss_type=config['loss'],
                learning_rate=config['lr'],
                log=log)

if __name__ == "__main__":
    # argsparse
    

    config = {
        # training params
        'model_name': 'GAT',
        'batch_size': 2,
        'epochs': 3,
        'lr': 0.01,
        'loss' :  'mse',
        'subset_size' : 10 ,

        # graph params
        'graph_layers': 3,
        'graph_hidden_channels': 256,
        'attention_heads': 4,
        'attention_dropouts': 0.2,
        'graph_dropouts': 0.2,
        'graph_norm': True,

        # predictor params
        'pred_layers': 3,
        'pred_hidden_channels': 256,
        'pred_dropouts': 0.2,
    }

    wandb.init(project="mrna_tests")
    run_training(config, log=True)