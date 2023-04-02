from ViennaRNA import RNA
import torch


def base_type_encoder(base: str) -> int:
    # Label encoder for basetypes, to ensure consistant labels.
    base_types = {
        "A": 0,
        "C": 1,
        "G": 2,
        "U": 3
    }

    return base_types[base]


class mRNAFeaturizer:
    """
    Creates node features, edges and attributes from fasta strings
    """

    def featurize(self, molecules):
        all_feats = []
        
        for molecule in molecules:

            edge_features = self.edge_features(molecule)

            feats = {
                "node_attributes": self.node_features(molecule),
                "edge_indexes": edge_features['edge_index'],
                "edge_attributes": edge_features['edge_attributes']
            }

            all_feats.append(feats)

        return all_feats

    def edge_features(self, sequence: str):
        # 2 dimensional edges [A,B]
        # A: 1 = pair along the sequence, 0 = not next to eachother
        # B: base pair probability

        # get bppm data for the mRNA molecule
        fc = RNA.fold_compound(sequence)
        (propensity, ensemble_energy) = fc.pf()
        bppm = fc.bpp()
        bppm = [list(t) for t in bppm]

        edge_indexes = []
        edge_attributes = []
        for i in range(0, len(sequence)):
            for j in range(0, len(sequence)):
                if i != j:
                    # add edges
                    edge_indexes.append([i, j])
                    # add edge weights as attributes
                    if i+1 == j:
                        edge_attributes.append([1, bppm[i][j]])
                    else:
                        edge_attributes.append([0, bppm[i][j]])

        return {'edge_index': edge_indexes,
                'edge_attributes': edge_attributes}

    def node_features(self, sequence: str):
        # just base type
        # TODO: alternatives?
        node_attr = [[base_type_encoder(t)] for t in sequence]

        return node_attr
