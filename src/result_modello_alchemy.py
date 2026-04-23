#import generale
import pandas as pd 
from parser.data_loader import Alchemy_parser
from pathlib import Path
from plots.model_visualizer import ModelVisualizer

#import math
from my_math.basic_math import MathBase

#import graph 
from modello.modello_GNN import GNNModel

from graph.graf import Graph_tools
#read csv as dataframe
df = pd.read_csv("../data/Alchemy-v20191129/final_version.csv")

#df.info()
#print(df.columns[4])  
#print(repr(df.columns.tolist()))

#obj
parser = Alchemy_parser()
#extractor

dati_sdf = {}
dati_sdf["molecule_id"] = []
dati_sdf["smiles"] = []

base = Path("../data/Alchemy-v20191129")
folders = ["atom_9", "atom_10", "atom_11", "atom_12"]

dati_sdf = {
    #stesso nome per l inner join
    "gdb_idx": [],
    "smiles": [],
    "folder": []
}

for folder_name in folders:
    folder = base / folder_name

    for sdf_file in folder.glob("*.sdf"):
        mol_id, smiles = parser.sdf_file_to_id_smiles(sdf_file)

        if smiles is not None:
            dati_sdf["gdb_idx"].append(int(mol_id))
            dati_sdf["smiles"].append(smiles)
            dati_sdf["folder"].append(folder_name)

df_sdf = pd.DataFrame(dati_sdf)

#print(df_sdf.info())

#print(df_sdf.head(10))

df_finale = parser.in_join(df_sdf,df)
print(df_finale.info())
print(df_finale.head(10))

#prediction

#obj graphhtols
graph_tools = Graph_tools()
datas = []

# smiles in Graph
for _, row in df_finale.iterrows():
    data = graph_tools.smiles_to_graph(row["smiles"])
    datas.append(data)

# Controlli rapidi sul secondo esempio
print(df_finale["smiles"].iloc[1])
print(datas[1])
print(datas[1].x.shape)
print(datas[1].edge_index.shape)
print(datas[1].y)


#def model path e device 
device = "cpu"
model_path = "../models/models.pth"
#obj modello non creato perche statitian mmethods no self
model = GNNModel.model_load(model_path, device=device) 

#predico nuovi valori 

preds = GNNModel.predict_new_gaps(model, datas, device=device, batch_size=32)

#metto preds dentro uan nuova colonna del dataframe

df_finale["gap_pred"] = preds

#print(df_finale.head(10))

#obj visualizzatore
visualizer = ModelVisualizer()

visualizer.plot_predictions(
    df_finale["gap"].to_numpy(),
    df_finale["gap_pred"],
    output_imm="../figures/pred_Alchimy_on_QM9model.png",
    title="Model for Alchimy : Predicted vs original in QM9"
)

visualizer.plot_errors(
    df_finale["gap"].to_numpy(),
    df_finale["gap_pred"].to_numpy(),
    output_imm="../figures/error_distribution_pred_alchimy_on_QM9_model.png",
    bins=10,
    title="Model  for alchimy"
)

math_base = MathBase()

mae, rmse, r2 = math_base.all_metrics(df_finale["gap"].to_numpy(), df_finale["gap_pred"].to_numpy())


df_atom_9 = df_finale[df_finale["folder"] == "atom_9"].reset_index(drop=True)
df_atom_10 = df_finale[df_finale["folder"] == "atom_10"].reset_index(drop=True)
df_atom_11 = df_finale[df_finale["folder"] == "atom_11"].reset_index(drop=True)
df_atom_12 = df_finale[df_finale["folder"] == "atom_12"].reset_index(drop=True)
print(df_atom_10.columns.to_list())

visualizer.plot_predictions(
    df_atom_9["gap"].to_numpy(),
    df_atom_9["gap_pred"].to_numpy(),
    output_imm="../figures/pred_Alchimy_on_QM9model_folder_9.png",
    title="Model for Alchimy : Predicted vs original in QM9"
)

mae, rmse, r2 = math_base.all_metrics(df_atom_9["gap"].to_numpy(), df_atom_9["gap_pred"].to_numpy())

visualizer.plot_predictions(
    df_atom_10["gap"].to_numpy(),
    df_atom_10["gap_pred"].to_numpy(),
    output_imm="../figures/pred_Alchimy_on_QM9model_folder_10.png",
    title="Model for Alchimy : Predicted vs original in QM9"
)

mae, rmse, r2 = math_base.all_metrics(df_atom_10["gap"].to_numpy(), df_atom_10["gap_pred"].to_numpy())

visualizer.plot_predictions(
    df_atom_11["gap"].to_numpy(),
    df_atom_11["gap_pred"].to_numpy(),
    output_imm="../figures/pred_Alchimy_on_QM9model_folder_11.png",
    title="Model for Alchimy : Predicted vs original in QM9"
)

mae, rmse, r2 = math_base.all_metrics(df_atom_11["gap"].to_numpy(), df_atom_11["gap_pred"].to_numpy())

visualizer.plot_predictions(
    df_atom_12["gap"].to_numpy(),
    df_atom_12["gap_pred"].to_numpy(),
    output_imm="../figures/pred_Alchimy_on_QM9model_folder_12.png",
    title="Model for Alchimy : Predicted vs original in QM9"
)

mae, rmse, r2 = math_base.all_metrics(df_atom_12["gap"].to_numpy(), df_atom_12["gap_pred"].to_numpy())



print(df_finale.groupby("folder")["gap"].agg(["min", "max", "mean", "std"]))


df_finale["has_S"] = df_finale["smiles"].str.contains("S")
df_finale["has_Cl"] = df_finale["smiles"].str.contains("Cl")

print(df_finale.groupby("folder")[["has_S", "has_Cl"]].mean())