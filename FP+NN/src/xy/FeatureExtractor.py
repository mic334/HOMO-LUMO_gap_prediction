import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs 


class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "mol_wt",
            "exact_mol_wt",
            "logp",
            "tpsa",
            "num_rings",
            "num_aromatic_rings",
            "num_rotatable_bonds",
            "num_h_donors",
            "num_h_acceptors",
            "num_heavy_atoms",
            "num_valence_electrons",
        ]

    def featurize_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return {
            "mol_wt": Descriptors.MolWt(mol),
            "exact_mol_wt": Descriptors.ExactMolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_h_donors": rdMolDescriptors.CalcNumHBD(mol),
            "num_h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
            "num_heavy_atoms": rdMolDescriptors.CalcNumHeavyAtoms(mol),
            "num_valence_electrons": Descriptors.NumValenceElectrons(mol),
        }

    def transform(self, df: pd.DataFrame):
        rows = []
        failed_smiles = []

        for _, row in df.iterrows():
            smiles = row["smiles"]
            features = self.featurize_smiles(smiles)

            if features is None:
                failed_smiles.append(smiles)
                continue

            rows.append({
                "molecule_id": row["molecule_id"],
                "smiles": smiles,
                "gap": row["gap"],
                **features
            })

        features_df = pd.DataFrame(rows)

        print(f"Righe processate: {len(features_df)}")
        print(f"SMILES falliti: {len(failed_smiles)}")

        return features_df, failed_smiles
    
    
        
    

    def get_xy(self, features_df: pd.DataFrame):
        X = features_df[self.feature_names].copy()
        y = features_df["gap"].copy()
        return X, y
    
    def compute_morgan_fingerprints(self, molecules, radius=2, n_bits=2048):
        fps = []
        for mol in molecules:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        return np.array(fps)