#import basic libraries
import pandas as pd     
import numpy as np
from pathlib import Path
#import math
from my_math.basic_math import  MathBase
#import file loading,savefile
from parser.data_loader import QM9Parser

def bench():

    output_filecsv = [
    ("../data/finale_elaborato/qm9_pred_test.csv", "../data/finale_elaborato/qm9_bench_test.csv"),
    ("../data/finale_elaborato/qm9_pred_train.csv", "../data/finale_elaborato/qm9_bench_train.csv")
    ]
    
    for output_filecsv, save_name in output_filecsv:
    #read CSV file 
        df = pd.read_csv(output_filecsv)

        print(df.head(10))
        #obj math
        m = MathBase()
        print("\n=== BENCHMARK GAP ===")
        #direct
        print("gap_pred_direct:")
        m.all_metrics(df["gap"],df["gap_pred_direct"])
        #influenced
        print("gap_pred_direct_influenced_HL:")
        m.all_metrics(df["gap"],df["gap_pred_direct_influenced_HL"])
        #HL
        print("gap_pred_HL:")
        m.all_metrics(df["gap"],df["gap_pred_HL"])

        df["diff_gap_influenced_HL"] = abs(
        df["gap_pred_HL"] - df["gap_pred_direct_influenced_HL"]
        )
        df_diff = df.sort_values(
        "diff_gap_influenced_HL",
        ascending=False
        )
        #print("\n=== TOP 10 MOST INFLUENCED BY HL ===")
        #print(df_diff.head(25))
        #print("\n=== TOP 10 LEAST INFLUENCED BY HL ===")
        #print(df_diff.tail(25))
        #obj to save csv
        parser = QM9Parser("../data/raw/qm9_dataset/finale_elaborato")
        
        #save csv with the new column diff_gap_influenced_HL 
        parser.save_csv(df_diff, save_name)
    
if __name__== "__main__":

    bench()