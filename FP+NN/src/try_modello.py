#dataset creation
#download from https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/

from html import parser
from xy.FeatureExtractor import FeatureExtractor
from parser.data_loader import QM9Parser
import pandas as pd
import numpy as np
from rdkit import Chem
import joblib 
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)



def try_modello():
    input_path = "../data/PUB_raw"
    output_file = "../data/PUB_processed/compound_data.csv"
    
    #obj creation   
    parser = QM9Parser(input_path)
    
    input_file = f"{parser.folder_path}/Compound_000000001_000500000.sdf"
    
    df = parser.extract_sdf_data(input_file)
    print(f"Shape del DataFrame after sampling: {df.shape[0]} row(s)")#devono essere due colonne: smiles e gap
    
    df = df.copy()
    df["molecule_id"] = range(len(df))
    df["gap"] = pd.NA 
    df = df[["molecule_id", "smiles","gap"]]
    print(f"DataFrame con molecule_id e smiles: {df.shape}")
    df.head(10)
    #salvataggio csv
    parser.save_csv(df,output_file)
    #print(f"CSV salvato in: {output_file}")
    print(df.columns.tolist())
    extractor = FeatureExtractor()
    
    features_df, error_df = extractor.transform(df)
    
    # SMILES -> molecules
    features_df["molecules"] = [Chem.MolFromSmiles(smiles) for smiles in features_df["smiles"]]
    features_df = features_df[features_df["molecules"].notnull()].reset_index(drop=True)
    
    # Morgan fingerprints
    X = extractor.compute_morgan_fingerprints(features_df["molecules"])
    print("Shape X (fingerprints):", X.shape)
    print(X[:5])  # Stampa le prime 5 righe delle fingerprint per verifica
    
    model = joblib.load("../models/mlp_model.pkl")
    scaler = joblib.load("../models/mlp_scaler.pkl")
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    print("Predizioni:", y_pred[:10])
    features_df["gap"] = y_pred
    print(features_df.head(10))
    result_df = features_df[["molecule_id", "smiles", "gap"]]
    result_df = result_df.sort_values(by="gap", ascending=True)
    parser.save_csv(result_df, "../data/PUB_processed/compound_predictions.csv")
    
    
    
if __name__ == "__main__":
    try_modello()