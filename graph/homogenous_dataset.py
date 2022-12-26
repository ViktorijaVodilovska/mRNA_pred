from ViennaRNA import RNA
import json
import torch
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


def to_pytorch_dataset(dataset: pd.DataFrame, target_labels: List[str], train=True) -> list:
  """
  Creates a list of torch.geometric.data.Data objects representing each row of RNA molecule data.
  """
  graphs = []
  
  # todo: extract
  le = LabelEncoder()

  # Iterate data
  for seq_index,row in dataset.iterrows():    
    # Create nodes

    # tranform node type (base) to numerical tensor
    targets = le.fit_transform(list(row['sequence']))
    node_attr = torch.FloatTensor([[t] for t in targets])
    # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    # Create edges

    # get bppm data for the mRNA molecule
    fc = RNA.fold_compound(row['sequence'])
    (propensity, ensemble_energy) = fc.pf()
    bppm = fc.bpp()
    bppm = [list(t) for t in bppm]

    # add each base pair probability as an edge
    edge_indexes = []
    edge_attributes = []
    for i in range(0, row['seq_length']):
      for j in range(0, row['seq_length']):
        if i!=j:
          # add edges
          edge_indexes.append([i, j])
          # add edge weights as attributes
          edge_attributes.append([bppm[i][j]])

    # create tensors
    edge_index = torch.tensor(edge_indexes, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    # Add y(ground truth) for each node -> seq_len x 5 dimensional vector (for the different degradation rates)
    outputs_per_node = []
    for node in range(0, row['seq_scored']):
      outputs = []
      if train:
        for l in target_labels:
          outputs.append(row[l][node])
      else:
        for l in target_labels:
          outputs.append(float(json.loads(row[l])[node]))

      outputs_per_node.append(outputs)
    y = torch.tensor(outputs_per_node, dtype=torch.float)

    # Create Data object for graph (molecule)
    if train:
      # create "train_mask" to denote the nodes with outputs
      ones = torch.ones(row['seq_scored'], dtype=torch.long)
      zeros = torch.zeros(row['seq_length'] - row['seq_scored'], dtype=torch.long)
      train_mask = torch.cat((ones, zeros), 0)
      graph = Data(x=node_attr, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y, train_mask=train_mask)
    else:
      graph = Data(x=node_attr, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)
    graphs.append(graph)

  return graphs