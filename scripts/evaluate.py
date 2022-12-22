import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


def test_model(loader, model, target_labels, hetero=False):
    """
    Test model on given test data and return column wise RMSE
    """
    model.eval()

    gt_df = pd.DataFrame(columns=target_labels)
    pred_df = pd.DataFrame(columns=target_labels)

    for data in loader:  # Iterate in batches over the training/test dataset.
        if hetero:
            preds = model(data.x_dict, data.edge_index_dict).detach().numpy()
            truths = data.y_dict['base'].detach().numpy()
        else:
            preds = model(data).detach().numpy()
            truths = data.y.detach().numpy()
        for node in range(len(truths)):
            pred_df = pd.DataFrame(
                [dict(zip(target_labels, preds[node])) for node in range(len(truths))])
            gt_df = pd.DataFrame(
                [dict(zip(target_labels, truths[node])) for node in range(len(truths))])

    res = dict()
    for c in gt_df.columns:
        res[f'mae_{c}'] = mean_absolute_error(gt_df[c], pred_df[c])
        res[f'mse_{c}'] = mean_squared_error(gt_df[c], pred_df[c])
        res[f'rms_{c}'] = sqrt(mean_squared_error(gt_df[c], pred_df[c]))

    res['mcrmse'] = np.mean([res[i] for i in res.keys() if 'rms' in i])
    return res
