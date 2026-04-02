import xy.FeatureExtractor  as FeatureExtractor
import xtb_DFT.Lettura_Scrittura as Lettura_Scrittura
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd 
from pathlib import Path


df = pd.read_csv("../data/PUB_processed/compound_predictions.csv")
df.info()
df_new = pd.concat([
    df[["molecule_id", "smiles", "gap"]].head(5),
    df[["molecule_id", "smiles", "gap"]].tail(5)
]).reset_index(drop=True)
print(df_new.head(10))

output_dir = Path("../data/PUB_processed/xyz_files")   


output_dir = Path("../data/PUB_processed/xyz_files")

for _, row in df_new.iterrows():
    out = Lettura_Scrittura.smiles_to_xyz(
        smiles=row["smiles"],
        mol_id=row["molecule_id"],
        output_dir=output_dir
    )
    if out is None:
        print(f"Saltata molecola {row['molecule_id']}")

