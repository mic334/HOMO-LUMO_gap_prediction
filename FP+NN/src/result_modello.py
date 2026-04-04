import xy.FeatureExtractor as FeatureExtractor
import xtb_DFT.Lettura_Scrittura as Lettura_Scrittura
from xtb_DFT.run_xtb import run_xtb
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
from pathlib import Path
import subprocess as sp
import os
import time

df = pd.read_csv("../data/PUB_processed/compound_predictions.csv")
df.info()

df_new = pd.concat([
    df[["molecule_id", "smiles", "gap"]].head(5),
    df[["molecule_id", "smiles", "gap"]].tail(5)
]).reset_index(drop=True)

print(df_new.head(10))

output_dir = Path("../data/PUB_processed/xyz_files")

# 1) prima crea tutti gli xyz
for _, row in df_new.iterrows():
    out = Lettura_Scrittura.smiles_to_xyz(
        smiles=row["smiles"],
        mol_id=row["molecule_id"],
        output_dir=output_dir
    )
    if out is None:
        print(f"Saltata molecola {row['molecule_id']}")

# 2) poi lancia xtb
os.environ["OMP_NUM_THREADS"] = "4"

path_xyz = "../data/PUB_processed/xyz_files"
runner_xtb = run_xtb(path_xyz)

xyz_files = [f for f in os.listdir(path_xyz) if f.endswith(".xyz")]

t_start = time.time()

for xyz_file in xyz_files:
    t0 = time.time()
    print(f"Parto con {xyz_file}")
    runner_xtb.xtb(xyz_file) 
    t1 = time.time()
    print(f"{xyz_file} finito in {t1 - t0:.2f} s")

t_end = time.time()
print(f"Tempo totale: {t_end - t_start:.2f} s")