#import basic libraries
from curses import window
import pandas as pd     
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
#import math
from my_math.basic_math import  MathBase
#import file loading,savefile
from parser.data_loader import QM9Parser

def bench_qm9():

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

        #migliore gap diretto influenzato da HL
        df["diff_gap_influenced_HL_vsreal"] = abs(df["gap"] - df["gap_pred_direct_influenced_HL"])
        
        df["diff_gap_influenced_HL_vsHL"] = abs( 
        df["gap_pred_HL"] - df["gap_pred_direct_influenced_HL"]
        )
        df_diff = df.sort_values(
        "diff_gap_influenced_HL_vsHL",
        ascending=False
        )
        
        
        
        df_plot = df_diff.sort_values("diff_gap_influenced_HL_vsHL").reset_index(drop=True)

        df_plot["diff_norm"] = df_plot["diff_gap_influenced_HL_vsHL"] / df_plot["diff_gap_influenced_HL_vsHL"].max()
        df_plot["err_norm"] = df_plot["diff_gap_influenced_HL_vsreal"] / df_plot["diff_gap_influenced_HL_vsreal"].max()

        window = 100

        plt.figure(figsize=(10, 5))

        plt.plot(
        df_plot["diff_norm"].rolling(window).mean(),
        label="Diff predizioni norm"
        )

        plt.plot(
        df_plot["err_norm"].rolling(window).mean(),
        label="Errore reale norm"
        )

        plt.xlabel("Molecole ordinate per disaccordo")
        plt.ylabel("Media mobile normalizzata")
        plt.title("Trend disaccordo vs errore reale")
        plt.legend()
        plt.grid(True)
        plt.show()
        
       #scatter plot di base 
        plt.figure(figsize=(8,6))

        plt.scatter(
        df_diff["diff_gap_influenced_HL_vsHL"],
        df_diff["diff_gap_influenced_HL_vsreal"],
        alpha=0.4,
        s=15
        )

        plt.xlabel("Disaccordo tra le due predizioni")
        plt.ylabel("Errore rispetto al valore reale")
        plt.title("Scatter plot: disaccordo vs errore reale")
        plt.grid(True)
        plt.show()
       
       
       
        
        
        
        
        
        #print("\n=== TOP 10 MOST INFLUENCED BY HL ===")
        #print(df_diff.head(25))
        #print("\n=== TOP 10 LEAST INFLUENCED BY HL ===")
        #print(df_diff.tail(25))
        #obj to save csv
        parser = QM9Parser("../data/raw/qm9_dataset/finale_elaborato")
        
        #save csv with the new column diff_gap_influenced_HL 
        parser.save_csv(df_diff, save_name)
    
if __name__== "__main__":

    bench_qm9()