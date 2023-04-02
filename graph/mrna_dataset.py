import pandas as pd
from typing import List
import torch
from torch_geometric.data import Dataset, Data, Batch
import json

from graph.featurizer import mRNAFeaturizer



# TODO: add heterograph option

class mRNADataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_cols: List[str],
        input_col: str = 'sequence',
        scored_nodes_col: str = 'seq_scored',
        number_nodes_col: str = 'seq_length'
    ) -> None:
        super().__init__()

        self.molecules = data[[input_col]]

        self.target_labels = target_cols
        self.target_values = data[target_cols]
        if scored_nodes_col != None:
            self.scored_lengths = data[scored_nodes_col] # TODO: make configurable?
        else:
            self.scored_lengths = [len(mol) for mol in self.molecules]
        if number_nodes_col != None:
            self.seq_lengths = data[number_nodes_col] # TODO: make configurable?
        else:
            self.seq_lengths = [len(mol) for mol in self.molecules]
        

        self.featurizer = mRNAFeaturizer()


    def len(self):
        return len(self.molecules)


    def get(self, idx):

        featurized = self.featurizer.featurize(self.molecules.loc[idx])
        outputs = self.get_targets_info(idx)

        data = Batch.from_data_list([
            Data(
                x=torch.Tensor(mol['node_attributes']),
                edge_index=torch.tensor(mol['edge_indexes'], dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(mol['edge_attributes'], dtype=torch.float),
                y=torch.tensor(output['targets'], dtype=torch.float),
                train_mask = output['masks']
            )
            for mol,output in zip(featurized, outputs)
        ])

        return data

    def get_targets_info(self, idx):
        if type(idx) == int:
            idx = [idx]

        outputs_per_graph = []
        
        for index in idx:
            outputs_per_node = []
            for node in range(0, self.scored_lengths[index]):
                outputs = []
                for l in self.target_labels:
                    try:
                        outputs.append(float(self.target_values[l][index][node]))
                    except:
                        outputs.append(float(list(json.loads(self.target_values[l][index]))[node]))
                        
                outputs_per_node.append(outputs)

            #TODO toggle train masks for train/test? or relabel as label_mask and use for eval too!
            ones = torch.ones(self.scored_lengths[index], dtype=torch.long)
            zeros = torch.zeros(self.seq_lengths[index] - self.scored_lengths[index], dtype=torch.long)
            train_mask = torch.cat((ones, zeros), 0)
            
            outputs_per_graph.append({'targets':outputs_per_node,'masks':train_mask})

        return outputs_per_graph


    def get_graph_info(self):

        graph_info = {
            "node_dim": self.get(0)[0].x.shape[1],
            "edge_dim": self.get(0)[0].edge_attr.shape[1],
        }

        return graph_info
