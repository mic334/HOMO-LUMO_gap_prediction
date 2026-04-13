import pandas as pd
import torch

from parser.data_loader import QM9Parser
from graph.graf import Graph_tools
from modello.modello_GNN import GNNModel
from html import parser

def try_modello():
    # Percorsi input/output e modello salvato
    input_path = "../data/PUB_raw"
    output_file = "../data/PUB_processed/compaund_data.csv"
    model_path = "../models/models.pth"
    device = "cpu"

    # Parser per leggere il file SDF
    parser = QM9Parser(input_path)
    input_file = f"{parser.folder_path}/Compound_000000001_000500000.sdf"

    # Estrae i dati dal file SDF in un DataFrame
    df = parser.extract_sdf_data(input_file)
    print(f"Shape del DataFrame after sampling: {df.shape[0]} row(s)")

    # Preparo il DataFrame con le colonne utili alla predizione
    df = df.copy()
    df["molecule_id"] = range(len(df))
    df["gap"] = pd.NA
    df = df[["molecule_id", "smiles", "gap"]]

    print(f"DataFrame con molecule_id e smiles: {df.shape}")
    print(df.head(10))

    # Salvo il csv processato
    parser.save_csv(df, output_file)
    print(df.columns.tolist())

    # Converto ogni SMILES in un grafo PyG
    graph_tools = Graph_tools()
    datas = []

    for _, row in df.iterrows():
        data = graph_tools.smiles_to_graph(row["smiles"])
        datas.append(data)

    # Controlli rapidi sul secondo esempio
    print(df["smiles"].iloc[1])
    print(datas[1])
    print(datas[1].x.shape)
    print(datas[1].edge_index.shape)
    print(datas[1].y)

    # Carico il modello già addestrato
    model = GNNModel.model_load(model_path, device=device)

    # Faccio la predizione dei gap sui nuovi grafi
    preds = GNNModel.predict_new_gaps(model, datas, device=device, batch_size=32)

    # Aggiungo le predizioni al DataFrame finale
    df["pred_gap"] = preds

    # Stampo un'anteprima del risultato
    print(df[["smiles", "pred_gap"]].head(10))
    df = df.sort_values(by="pred_gap", ascending=True)
    parser.save_csv(df,"../data/PUB_processed/compound_predictions.csv")

if __name__ == "__main__":
    try_modello()