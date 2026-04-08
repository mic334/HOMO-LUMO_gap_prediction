import xy.FeatureExtractor as FeatureExtractor
from plots.model_visualizer import ModelVisualizer
import xtb_DFT.Lettura_Scrittura as Lettura_Scrittura
from xtb_DFT.run_xtb import run_xtb
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
from pathlib import Path
import subprocess as sp
from io import StringIO
import os
import time
#import resource

df = pd.read_csv("../data/PUB_processed/compound_predictions.csv")
df.info()
'''
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

# stack (equivalente a ulimit -s unlimited)
#resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"      # parti da 1, poi eventualmente 4
env["OMP_STACKSIZE"] = "4G"
# variabili ambiente

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
'''
result = sp.run(
    ["bash", "../script/estrai_HOMO_LUMO_xtb.sh"],
    capture_output=True,
    text=True,
    check=True
)

print("parto da qua")

gap_calcolati_ev = pd.read_csv(StringIO(result.stdout))
gap_calcolati_H = gap_calcolati_ev.copy()
gap_calcolati_ev= gap_calcolati_ev.sort_values("id").reset_index(drop=True)
print(gap_calcolati_ev)
gap_calcolati_H["gap"] = gap_calcolati_ev["gap_ev"]/27.2114

print(gap_calcolati_ev.columns)
# prendo da df i valori del modello e li allineo agli id calcolati
df_plot = df[["molecule_id", "gap"]].rename(columns={"molecule_id": "id"}).copy()

df_plot = df_plot.merge(
    gap_calcolati_H[["id", "gap"]],
    on="id",
    how="inner",
    suffixes=("_pred", "_real")
).sort_values("id").reset_index(drop=True)

print(df_plot)

visualizer = ModelVisualizer()

visualizer.plot_predictions(
    df_plot["gap_real"].to_numpy(),
    df_plot["gap_pred"].to_numpy(),
    output_imm="../figures/pred_vs_calcc" ,
    title="Model vs XTB: Predicted vs calc"
)

visualizer.plot_errors(
    df_plot["gap_real"].to_numpy(),
    df_plot["gap_pred"].to_numpy(),
    output_imm= "../figures/error_distribution_xtb.png",
    bins=10,
    title="Model vs XTB: Error Distribution"
)