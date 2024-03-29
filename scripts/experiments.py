import argparse
import json
import os
from pathlib import Path
import torch
from model.predictor import Predictor
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Any
from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
import wandb
from graph.mrna_dataset import mRNADataset



def get_graph_info(example, hetero=False):
    # TODO: PUT IN DATASET CLASS
    graph_info = {}
    if hetero:
        graph_info['metadata'] = example.metadata()
        graph_info['in_channels'] = example.num_node_features
    else:
        graph_info['node_dim'] = example.x.shape[1]

    return graph_info


def run_training(config: Dict[str, Any], log: bool = True, save: bool = False, test: bool = False):
    """
    _summary_

    Args:
        config (Dict[str, Any]): _description_
        log (bool, optional): _description_. Defaults to True.
        save (bool, optional): _description_. Defaults to False.
        test (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    print(config)
    # TODO: add loading from checkpoint?

    # save experiment settings
    if save:
        with open(settings.get_model_folder(config['config_name']) / "config.json", "w") as f:
            json.dump(config, f)

    hetero_data = config['model_name'] == 'HAN'

    # Load train and val data
    # TODO: PUT IN SEPERATE FUNC
    data = pd.read_json(settings.TRAIN_DATA, lines=True).sample(
        frac=config['subset_size'], random_state=0)
    train, val = train_test_split(data, test_size=0.2)
    train=train.reset_index()
    val=val.reset_index()
    
    train_dataset = mRNADataset(data = train, target_cols = settings.TARGET_LABELS)
    val_dataset = mRNADataset(data = val, target_cols = settings.TARGET_LABELS)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)


    # make model
    print("First entry example:")
    print(train_dataset[0])
    graph_info = get_graph_info(train_dataset[0], hetero_data)

    pred_model = Predictor.from_config(config, graph_info)

    if log:
        wandb.init(project=settings.PROJECT_NAME,
                   config=config, name=config['config_name'])
        
    # train model
    res, model = train_model(pred_model,
                             train_loader,
                             val_loader,
                             config['epochs'],
                             settings.TARGET_LABELS,
                             loss_type=config['loss'],
                             learning_rate=config['lr'],
                             hetero=hetero_data,
                             log=log,
                             save_to=settings.get_model_path(config['config_name']) if 'config_name' in config  else None)

    # test model
    if test:
        res = run_testing(settings.get_model_folder(
            config['config_name']), log)

    return res, model


def run_testing(model_folder: Path, log: bool = True):
    """_summary_

    Args:
        model_path (Path): _description_
        log (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    # load config
    with open(model_folder / 'config.json', 'r') as f:
        config = json.load(f)

    hetero_data = config['model_name'] == 'HAN'

    # TODO: add hetero in dataset class  
    # load test data
    test = pd.read_csv(settings.TEST_DATA)
    test_dataset = mRNADataset(data = test, target_cols = settings.TEST_LABELS)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    # get graph info
    graph_info = get_graph_info(test_dataset[0], hetero_data)

    # load model
    model = Predictor.from_config(config, graph_info)
    model.load_state_dict(torch.load(model_folder / 'model.pth'))

    # test model
    res = test_model(test_loader, model, settings.TEST_LABELS)

    # log results
    if log:
        wandb.log({'evaluation': res})

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing argument")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to json file with an experiment configuration")
    parser.add_argument("--log", type=bool, default=False,
                        help="Log to wandb?")
    parser.add_argument("--train", type=bool,
                        default=False, help="Train model?")
    parser.add_argument("--test", type=bool, default=False, help="Test model?")
    parser.add_argument("--save", type=bool, default=False,
                        help="Save model after training?")
    args = parser.parse_args()

    if args.config_path == None:
        # example config
        # BEST GAT FROM TUNING
        config_path = settings.EXPERIMENTS_FOLDER / 'full_expert_gat/config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open(args.config_path, 'r') as f:
            config = json.load(f)

    if args.train:
        res, model = run_training(config, args.log, args.save, args.test)
        print(res)

    elif args.test:
        res = run_testing(settings.get_model_folder(
            config['config_name']), args.log)
        print(res)
