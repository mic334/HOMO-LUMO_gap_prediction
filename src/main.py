import pandas as pd
from parser.data_loader import QM9Parser 
from plots.model_visualizer import ModelVisualizer
from graph.graf import Graph_tools
from modello.modello_GNN import GNNModel, GNNTrainer
from torch_geometric.loader import DataLoader

def main():
    input_path = "../data/raw/qm9_dataset"
    output_path = "../data/processed/qm9_gap_dataset.csv"

    parser = QM9Parser(input_path)
    print(parser.folder_path)

    df = parser.parse_folder()
    parser.save_csv(df, output_path)
    
    #extraction for X[descriptors(molecule_id,rings,weights,ecc)] Y[molecule_id,gap]
    df = pd.read_csv(output_path)
    print(f"Dataset caricato: {len(df)} righe")
    df.info()
    #print(df.head(10))
    
    #obj
    graph = Graph_tools()
    
    datas= []
    for _,row in df.iterrows():
        smile = row["smiles"]
        target = row["gap"]
        
        data= graph.smiles_to_graph(smile,target)
        datas.append(data)
    print(df["smiles"].iloc[1])
    print(datas[1])
    print(datas[1])
    print(datas[1].x.shape)
    print(datas[1].edge_index.shape)
    print(datas[1].y)
    
    #auto test size 0.2 random seed 42
    train_data, test_data = graph.split_data_GNN(datas, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    #model creation obj 
    
    model = GNNModel(in_channels=8,hidden_channels=128)
    
    #ogetto trainer in per poi usare run(funzione che fa il training )
    trainer = GNNTrainer(model, train_loader, test_loader, lr=0.001)
    print(trainer.device)
    trainer.run(epochs=50)
    
    #save modello
    path_modello = "../models"
    model_name = "GNN.pth"
    trainer.save_model_gnn(model,model_name,path_modello)
    
    visualizer = ModelVisualizer()
    
    visualizer.plot_losses(trainer.train_losses, trainer.test_losses)
    visualizer.plot_epoch_times(trainer.epoch_time)
    
    y_true, y_pred = trainer.get_predictions()


    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_errors(y_true, y_pred)

    trainer.evaluate_GNN()

    print("y_true mean/std:", y_true.mean(), y_true.std())
    print("y_pred mean/std:", y_pred.mean(), y_pred.std())
    print("y_true min/max:", y_true.min(), y_true.max())
    print("y_pred min/max:", y_pred.min(), y_pred.max())

if __name__ == "__main__":
    main()
    
    