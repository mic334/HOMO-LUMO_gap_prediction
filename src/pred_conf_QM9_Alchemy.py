#import basic
from xml.parsers.expat import model
import pandas as pd 
import numpy as np
from pathlib import Path


#import models
from modello.modello_GNN import GNNModel
from graph.graf import Graph_tools


#import my things
from parser.data_loader import Alchemy_parser
from my_math.basic_math import MathBase

def __pred_conf_QM9_Alchemy():
    
    #def path
    data_path_alchemy= "../data/Alchemy-v20191129/final_version.csv"
    
    
    
    #load dataset alchimhy obj pd
    df = pd.read_csv(data_path_alchemy)
    print(df.shape[0])
    print("dati caricati")
    
    #obj parser for alchemy
    parser = Alchemy_parser()
    
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
    #print(df_finale.info()) 
    #print(df_finale.head(10))
    # OK,     gdb_idx                          smiles  folder       gap
    #      0  2536808          C[C@]1([NH3+])CC=CCCC1  atom_9  0.146845

    #prediction and conf 
    
    #obj  tools
    graph = Graph_tools()
    
    datas = []
    
    # smiles in Graph
    for _, row in df_finale.iterrows():
        data = graph.smiles_to_graph(row["smiles"])
        datas.append(data)

    # Controlli rapidi sul secondo esempio
    print(df_finale["smiles"].iloc[1])
    print(datas[1])
    print(datas[1].x.shape)
    print(datas[1].edge_index.shape)
    print(datas[1].y)


    #def model path e device 
    device = "cpu"
    model_path_gap = "../models/modello_gap.pth" 
    
    #obj modello gap diretto 
    model_gap = GNNModel.model_load(model_path_gap, device=device)
    #rediction 
    preds_direct_gap = GNNModel.predict_new_gaps(model_gap, datas, device=device, batch_size=32)
    #put in a datframe in a new colunm

    df_finale["gap_pred_direct"] = preds_direct_gap

    print(df_finale.head(10))

    #def modello 
    model_path_HL = "../models/modello_homo_lumo.pth"
    model_HL = GNNModel.model_load(model_path_HL, device=device)
    
    homo_preds, lumo_preds = GNNModel.predict_new_homo_lumo(model_HL, datas, device=device)
    
    #ad pred value for homo e lumo
    df_finale["Homo_pred"]= homo_preds
    df_finale["Lumo_pred"]= lumo_preds
    
    df_finale["gap_pred_HL"] = df_finale["Lumo_pred"] - df_finale["Homo_pred"]
    df_finale["abs_diff_gap"] = (df_finale["gap_pred_HL"] - df_finale["gap_pred_direct"]).abs()
    #print(df_finale.head(10))
    df_finale["err_direct"] = df_finale["gap"] - df_finale["gap_pred_direct"]
    df_finale["err_HL"] = df_finale["gap"] - df_finale["gap_pred_HL"]

    df_finale["abs_err_direct"] = df_finale["err_direct"].abs()   
    df_finale["abs_err_HL"] = df_finale["err_HL"].abs()

    df_finale["direct_better"] = df_finale["abs_err_direct"] < df_finale["abs_err_HL"]
    df_finale["HL_better"] = df_finale["abs_err_HL"] < df_finale["abs_err_direct"] 
    print(df_finale.head(10))
    
    
    
    #diff
    print(df[["abs_err_direct", "abs_err_HL"]].mean())
    print(df["direct_better"].mean(), df["HL_better"].mean())
    
    

if __name__ == "__main__":
    __pred_conf_QM9_Alchemy()