import os
import wandb
import torch
from pathlib import Path
from config import TrainConfig as train_settings
from scripts.evaluate import test_model


def train_model(model, train_loader, val_loader, epochs, target_labels, save_to: Path, loss_type: str = 'mse', hetero=False):
    criterion = train_settings.loss_function[loss_type]
    # changeable ?
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(epochs):
        
        losses = []
        
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            
            print(data.x.size(0))
            print(data.x)
            print(data.edge_index)
            
            out = model(data.x, data.edge_index)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            losses.append(loss)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        
        res = test_model(val_loader, model, target_labels, hetero=hetero)
        res['epoch'] = epoch
        res['epoch_loss'] = sum(losses)/len(losses)
        
        print(res)
        # wandb.log(res)
        
        # check before saving
        torch.save(model.state_dict(), save_to)
        
    return res
    