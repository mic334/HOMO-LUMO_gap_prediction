
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


class Graph_tools : 
    def __init__(self):
        pass
    
    def atom_features(self, atom):
        return [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
        ]
    def smiles_to_graph(self,smiles,target):
        mol = Chem.MolFromSmiles(smiles)    
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        x = [self.atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(x, dtype=torch.float)

        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        y = torch.tensor([target], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, y=y)
    
    def split_data_GNN(self, datas ,test_size=0.2, random_state=42):
        self.train_data, self.test_data = train_test_split(
            datas,
            test_size=test_size,
            random_state=random_state
        )

        print("Train size:", len(self.train_data))
        print("Test size :", len(self.test_data))
        return  self.train_data , self.test_data