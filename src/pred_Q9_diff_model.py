#import basic
from xml.parsers.expat import model
import pandas as pd 
import numpy as np
from pathlib import Path

from torch import device


#import models
from modello.modello_GNN import GNNModel
from graph.graf import Graph_tools


#import my things
from parser.data_loader import Alchemy_parser,QM9Parser
from my_math.basic_math import MathBase


def pred_Q9_diff_model():

    #set VARIABLE
    DEVICE = "cpu"
    BATCH_SIZE = 32

    #set path
    output_filecsv = "../data/processed/qm9_gap_HL_dataset.csv"
    output_path_model = "../models"
    name_model_gap = "modello_gap"
    name_model_homo_lumo = "modello_homo_lumo"
    
    #read a csv
    df =  pd.read_csv(output_filecsv)
    print("dati caricati")
    print(df.columns.to_list())
    
    #obj  tools
    graph = Graph_tools()
    
    
    model_path_gap = "../models/modello_gap.pth" 

#------datas----
    datas = []
    # smiles in Graph
    for _, row in df.iterrows():
        data = graph.smiles_to_graph(row["smiles"])
        datas.append(data)
#---------------direct gap prediction---------------------#    
    #obj modello gap diretto 
    model_gap = GNNModel.model_load(model_path_gap, device=DEVICE)
    #rediction 
    preds_direct_gap = GNNModel.predict_new_gaps(model_gap, datas, device=DEVICE, batch_size=BATCH_SIZE)
    #put in a datframe in a new colunm

    df["gap_pred_direct"] = preds_direct_gap

    
    #---------------hl prediction---------------------#
    #def modello 
    model_path_HL = "../models/modello_homo_lumo.pth"
    model_HL = GNNModel.model_load(model_path_HL, device=DEVICE)
    
    homo_preds, lumo_preds = GNNModel.predict_new_homo_lumo(model_HL, datas, device=DEVICE)
    
    
        #ad pred value for homo e lumo
    df["Homo_pred"]= homo_preds
    df["Lumo_pred"]= lumo_preds
    
#---------------gap direct hl influenced prediction---------------------#
    #def modello 
    model_path_direct_gap_influenced_HL = "../models/direct_gap_influenced_HL.pth"
    model_gap_influenced_HL = GNNModel.model_load(model_path_direct_gap_influenced_HL, device=DEVICE)
    
    #two exit gap_inf   _ doublevector with homo Lumo (not saved)
    gap_influenced, _ = GNNModel.predict_new_homo_lumo(model_gap_influenced_HL, datas, device=DEVICE) 

    df["gap_pred_direct_influenced_HL"] = gap_influenced

#---------------build data_frame_for_analisys---------#

    df["gap_pred_HL"] = df["Lumo_pred"] - df["Homo_pred"]
    
    
    #print dataframe
    print(df.head(10))

    #cvs only with HOMO,LUMO,GAP
    df = df.drop(columns=['norm_dipole_moment', 'norm_static_polarizability'])
    
    df.info()
    
    parser_QM9 = QM9Parser("../data/finale_elaborato")
    
    parser_QM9.save_csv(df,"../data/finale_elaborato/qm9_pred.csv")
    

if __name__ == "__main__":
    pred_Q9_diff_model()