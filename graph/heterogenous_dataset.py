from ViennaRNA import RNA
import json
import torch
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData


def to_pytorch_heterodata(dataset: pd.DataFrame, target_labels: List[str], train=True) -> list:
    """
    Creates a list of torch.geometric.data.Data objects representing each row of RNA molecule data.
    """
    graphs = []

    # todo: extract
    le = LabelEncoder()

    # Iterate data
    for seq_index, row in dataset.iterrows():
        # Create nodes

        # tranform node type (base) to numerical tensor
        targets = le.fit_transform(list(row['sequence']))
        node_attr = torch.FloatTensor([[t] for t in targets])

        # Create edges
        sequential_edge_indexes = []
        sequential_edge_attributes = []
        # add an edge for each sequential connection (phosphodiester bonds)
        for i in range(0, row['seq_length']-1):
            sequential_edge_indexes.append([i, i+1])
            sequential_edge_attributes.append([1])

        # get bppm data for the mRNA molecule
        fc = RNA.fold_compound(row['sequence'])
        (propensity, ensemble_energy) = fc.pf()
        bppm = fc.bpp()
        bppm = [list(t) for t in bppm]

        # add each base pair probability as an edge
        bppm_edge_indexes = []
        bppm_edge_attributes = []
        for i in range(0, row['seq_length']):
            for j in range(0, row['seq_length']):
                if i != j:
                    # add edges
                    bppm_edge_indexes.append([i, j])
                    # add edge weights as attributes
                    bppm_edge_attributes.append([bppm[i][j]])

        # create tensors
        sequential_edge_index = torch.tensor(
            sequential_edge_indexes, dtype=torch.long)
        sequential_edge_attr = torch.tensor(
            sequential_edge_attributes, dtype=torch.long)
        bppm_edge_index = torch.tensor(bppm_edge_indexes, dtype=torch.long)
        bppm_edge_attr = torch.tensor(bppm_edge_attributes, dtype=torch.float)

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

        # Create HeteroData object for graph (molecule)
        graph = HeteroData()
        graph['base'].x = node_attr
        graph['base'].y = y
        graph['base', 'phosphodiester_bond', 'base'].edge_index = sequential_edge_index.t().contiguous()
        graph['base', 'phosphodiester_bond', 'base'].edge_attr = sequential_edge_attr
        graph['base', 'hydrogen_bond', 'base'].edge_index = bppm_edge_index.t().contiguous()
        graph['base', 'hydrogen_bond', 'base'].edge_attr = bppm_edge_attr

        # Create Data object for graph (molecule)
        if train:
            # create "train_mask" to denote the nodes with outputs
            ones = torch.ones(row['seq_scored'], dtype=torch.long)
            zeros = torch.zeros(row['seq_length'] -
                                row['seq_scored'], dtype=torch.long)
            train_mask = torch.cat((ones, zeros), 0)
            graph.train_mask = train_mask

        graphs.append(graph)

    return graphs
