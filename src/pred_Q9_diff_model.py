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
    
    
    


    


if __name__ == "__main__":
    pred_Q9_diff_model()