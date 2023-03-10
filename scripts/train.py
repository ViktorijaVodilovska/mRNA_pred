import os
import wandb
import torch
from pathlib import Path
from config import TrainConfig as train_settings, BaseSettings
from scripts.evaluate import test_model


def train_model(model, train_loader, val_loader, epochs, target_labels, save_to: Path = None, loss_type: str = 'mse', learning_rate: float = 0.01, hetero=False, log=True):
    criterion = train_settings.loss_function[loss_type]()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if log:
        wandb.init(project=BaseSettings.PROJECT_NAME)
    
    for epoch in range(epochs):
        
        losses = []
        
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            out = model(data)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            losses.append(loss.detach().numpy())
        
        res = test_model(val_loader, model, target_labels, hetero=hetero)
        res['epoch'] = epoch
        epoch_loss = sum(losses)/len(losses) if len(losses)>0 else 0
        print(epoch_loss)
        res['epoch_loss'] = epoch_loss
        print(res)
        if log:
            wandb.log(res)
        
        # check before saving
        if save_to != None:
            torch.save(model.state_dict(), save_to)
        
    return res
    