import pandas as pd 
import numpy as np


def new_train():
    
    df = pd.read_csv("../data/finale_elaborato/qm9_bench.csv")
    
    #print the first 10 rows of the dataframe 
    df.head(10)

    #pool of old molecules to train 
    top_50 = df.head(50).reset_index(drop=True)
    top_100 = df.head(100).reset_index(drop=True)
    top_250 = df.head(250).reset_index(drop=True)

    if __name__== "__main__":
        new_train() 