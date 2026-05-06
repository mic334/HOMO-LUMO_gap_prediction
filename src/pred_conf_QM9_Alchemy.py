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
    
    DEVICE = "cpu"
    BATCH_SIZE = 32
    
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
    device = DEVICE
    model_path_gap = "../models/modello_gap.pth" 

#---------------direct gap prediction---------------------#    
    #obj modello gap diretto 
    model_gap = GNNModel.model_load(model_path_gap, device=device)
    #rediction 
    preds_direct_gap = GNNModel.predict_new_gaps(model_gap, datas, device=device, batch_size=BATCH_SIZE)
    #put in a datframe in a new colunm

    df_finale["gap_pred_direct"] = preds_direct_gap

    print(df_finale.head(10))
    
#---------------hl prediction---------------------#
    #def modello 
    model_path_HL = "../models/modello_homo_lumo.pth"
    model_HL = GNNModel.model_load(model_path_HL, device=device)
    
    homo_preds, lumo_preds = GNNModel.predict_new_homo_lumo(model_HL, datas, device=device)
    
    
        #ad pred value for homo e lumo
    df_finale["Homo_pred"]= homo_preds
    df_finale["Lumo_pred"]= lumo_preds
    
#---------------gap direct hl influenced prediction---------------------#
    #def modello 
    model_path_direct_gap_influenced_HL = "../models/direct_gap_influenced_HL.pth"
    model_gap_influenced_HL = GNNModel.model_load(model_path_direct_gap_influenced_HL, device=device)
    
    #two exit gap_inf   _ doublevector with homo Lumo (not saved)
    gap_influenced, _ = GNNModel.predict_new_homo_lumo(model_gap_influenced_HL, datas, device=device) 

    df_finale["gap_pred_direct_influenced_HL"] = gap_influenced

#---------------build data_frame_for_analisys---------#

    df_finale["gap_pred_HL"] = df_finale["Lumo_pred"] - df_finale["Homo_pred"]
    
    
    #print dataframe
    print(df_finale.head(10))

    # Differenza tra predizione direct e predizione HL
    df_finale["abs_diff_gap"] = (df_finale["gap_pred_HL"] - df_finale["gap_pred_direct"]).abs()

    # Errori firmati
    df_finale["err_direct"] = df_finale["gap"] - df_finale["gap_pred_direct"]
    df_finale["err_HL"] = df_finale["gap"] - df_finale["gap_pred_HL"]
    df_finale["err_influenced"] = (df_finale["gap"] - df_finale["gap_pred_direct_influenced_HL"])

    # Errori assoluti
    # Più piccolo = meglio
    df_finale["abs_err_direct"] = df_finale["err_direct"].abs()
    df_finale["abs_err_HL"] = df_finale["err_HL"].abs()
    df_finale["abs_err_influenced"] = df_finale["err_influenced"].abs()

    #--------------------fast analysis---------------------#
    # Chi vince molecola per molecola
    df_finale["direct_better"] = (
        (df_finale["abs_err_direct"] < df_finale["abs_err_HL"]) &
        (df_finale["abs_err_direct"] < df_finale["abs_err_influenced"])
    )

    df_finale["HL_better"] = (
        (df_finale["abs_err_HL"] < df_finale["abs_err_direct"]) &
        (df_finale["abs_err_HL"] < df_finale["abs_err_influenced"])
    )

    df_finale["influenced_better"] = (
        (df_finale["abs_err_influenced"] < df_finale["abs_err_direct"]) &
        (df_finale["abs_err_influenced"] < df_finale["abs_err_HL"])
    )


    
    
    print("\n=== ERRORE MEDIO ===")
    print("# più basso = meglio")

    print("direct     :", df_finale["abs_err_direct"].mean())
    print("HL         :", df_finale["abs_err_HL"].mean())
    print("influenced :", df_finale["abs_err_influenced"].mean())


    print("\n=== PERCENTUALE VITTORIE ===")
    print("# più alto = meglio")

    print("direct     :", df_finale["direct_better"].mean() *100, "%" )
    print("HL         :", df_finale["HL_better"].mean()*100,"%")
    print("influenced :", df_finale["influenced_better"].mean() * 100, "%")


if __name__ == "__main__":
    __pred_conf_QM9_Alchemy()