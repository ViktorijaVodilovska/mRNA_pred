import pandas as pd
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
      preds = model(data.x, data.edge_index).detach().numpy() 
      truths = data.y.detach().numpy() 
    for node in range(len(truths)):
        pred_df = pred_df.append(dict(zip(target_labels, preds[node])), ignore_index=True)
        gt_df = gt_df.append(dict(zip(target_labels, truths[node])),  ignore_index=True)
    
  res = dict(zip(['mae','mse','rms'], [{},{},{}]))
  for c in gt_df.columns:
    res['mae'][c] = mean_absolute_error(gt_df[c], pred_df[c])
    res['mse'][c] = mean_squared_error(gt_df[c], pred_df[c])
    res['rms'][c] = sqrt(mean_squared_error(gt_df[c], pred_df[c]))
  
  return res