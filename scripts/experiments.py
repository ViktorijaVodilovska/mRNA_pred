import os
import wandb
from model.gat import GAT
from model.predictor import Predictor
from torch_geometric.loader import DataLoader
from graph.homogenous_dataset import to_pytorch_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from config import BaseSettings as settings
from scripts.train import train_model

if __name__ == "__main__":
    
    # wandb.init(project="mrna_tests")
    
    # make run experiment func
    
    # put in config
    BATCH_SIZE = 1
    EPOCHS = 1
    LOSS_TYPE = 'mse'
    
    # put in config settings
    experiment_path = Path("experiments/test1/")
    model_path = experiment_path / "model.pth"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    
    # Load train data
    data = pd.read_json(settings.TRAIN_DATA, lines=True)[:10]
    
    train, val = train_test_split(data, test_size=0.2)
    
    train_dataset = to_pytorch_dataset(train, settings.BPPM_FOLDER, settings.TARGET_LABELS)
    val_dataset = to_pytorch_dataset(train, settings.BPPM_FOLDER, settings.TARGET_LABELS)
    print("First entry example:")
    print(train_dataset[0])
    
    NODE_DIM = train_dataset[0].x.shape[0]
    EDGE_DIM = train_dataset[0].edge_attr.shape[0]

    # make a data loader for batches 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # make model
    graph_model = GAT(node_dim=NODE_DIM, 
                      edge_dim=EDGE_DIM, 
                      layers=3, 
                      hidden_channels=[256, 256, 256], 
                      attention_heads=[8,8,8], 
                      attention_dropouts=[0.5,0.5,0.5], 
                      dropouts=[0.2,0.2,0.2], 
                      graph_norm=True)
    
    pred_model = Predictor(graph_model=graph_model,
                           layers=3, 
                           hidden_channels=[256,256],
                           output_channels=5,
                           dropouts=[0.2,0.2])
        
    train_model(pred_model, train_loader, val_loader, EPOCHS, settings.TARGET_LABELS, model_path)