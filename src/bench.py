import pandas as pd 
import numpy as np
from my_math.basic_math import  MathBase

def bench():

    #read CSV file 
    df = pd.read_csv("../data/finale_elaborato/qm9_pred.csv")

    print(df.head(10))
    
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


if __name__== "__main__":
    bench()