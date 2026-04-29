#import python libraries
import pandas as pd
import numpy as np

#import ia lib 
from torch_geometric.loader import DataLoader

#importing my models
from modello.modello_GNN import  GNNModel, GNNTrainer

#importing my class
from parser.data_loader import QM9Parser
from graph.graf import Graph_tools
from plots.model_visualizer import ModelVisualizer

def build_model():
    
    #def path
    input_path = "../data/raw/qm9_dataset"
    output_filecsv = "../data/processed/qm9_gap_HL_dataset.csv"
    output_path_model = "../models"
    name_model_gap = "modello_gap"
    name_model_homo_lumo = "modello_homo_lumo"
    output_imm = "../results_models/"

  
    #obj parser 
    parser = QM9Parser(input_path)
    print(parser.folder_path)
    df = parser.parse_folder()
    parser.save_csv(df, output_filecsv)

    #extraction for X[descriptors(molecule_id,rings,weights,ecc)] Y[molecule_id,gap]
    df = pd.read_csv(output_filecsv)
    print(f"Dataset caricato: {len(df)} righe")
    #info_csv_file
    df.info()
    
    #obj graph_tools
    graph = Graph_tools()
    
    
    #model for gap 
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
    
    train_data, test_data = graph.split_data_GNN(datas, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    #model creation obj 
    
    model_g = GNNModel(in_channels=8,hidden_channels=128)
    
    #ogetto trainer in per poi usare run(funzione che fa il training )
    trainer= GNNTrainer(model_g, train_loader, test_loader, lr=0.001)
    print(trainer.device)
    trainer.run(epochs=10)
    
    #save model gap 
    trainer.save_model_gnn(model_g,name_model_gap,output_path_model)
    
    #obj for plots
    visualizer = ModelVisualizer()
    
    #commetted loss
    #visualizer.plot_losses(trainer.train_losses, trainer.test_losses)
    
    #commented epoch_times
    #visualizer.plot_epoch_times(trainer.epoch_time)
    
    #use of model 
    y_true, y_pred = trainer.get_predictions()


    visualizer.plot_predictions(y_true, y_pred,output_imm + "gap.png", title="Predictions")
    visualizer.plot_errors(y_true, y_pred,output_imm + "error_gap.png", bins=20, title="errors")

    trainer.evaluate_GNN()

    print("y_true mean/std:", y_true.mean(), y_true.std())
    print("y_pred mean/std:", y_pred.mean(), y_pred.std())
    print("y_true min/max:", y_true.min(), y_true.max())
    print("y_pred min/max:", y_pred.min(), y_pred.max())
    
    #---------end gap --------------#

    #model for HL
    datas= []
    for _,row in df.iterrows():
        smile = row["smiles"]
        target = row["homo"] , row["lumo"]
        data= graph.smiles_to_graph(smile,target)
        datas.append(data)
    print(df["smiles"].iloc[1])
    print(datas[1])
    print(datas[1])
    print(datas[1].x.shape)
    print(datas[1].edge_index.shape)
    print(datas[1].y)
    
    train_data, test_data = graph.split_data_GNN(datas, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    #model creation obj 
    model_HL= GNNModel(in_channels=8,hidden_channels=128,out_channels=2,positive_output=False)
    
    #ogetto trainer in per poi usare run(funzione che fa il training )
    trainer = GNNTrainer(model_HL, train_loader, test_loader, lr=0.001)
    print(trainer.device)
    trainer.run(epochs=10)    

    #save model HOMO LUMO 
    trainer.save_model_gnn(model_HL,name_model_homo_lumo,output_path_model)
    visualizer = ModelVisualizer()
    
    #commented loss
    #visualizer.plot_losses(trainer.train_losses, trainer.test_losses)
    #commeted epochs
    #visualizer.plot_epoch_times(trainer.epoch_time)
    
    #prediction 
    y_true, y_pred = trainer.get_predictions()

    #homo part vector
    y_true_homo = y_true[:, 0]
    y_pred_homo = y_pred[:, 0]

    #lumo part vector
    y_true_lumo = y_true[:, 1]
    y_pred_lumo = y_pred[:, 1]

    
    visualizer.plot_predictions(y_true_homo, y_pred_homo,output_imm + "H.png" , title="Predictions")
    visualizer.plot_errors(y_true_homo, y_pred_homo,output_imm + "H_errors.png", bins=20, title="errors")
    visualizer.plot_predictions(y_true_lumo, y_pred_lumo,output_imm + "L.png", title="Predictions_L")
    visualizer.plot_errors(y_true_lumo, y_pred_lumo,output_imm + "L_errors.png", bins=20, title="errors_L")

     
    trainer.evaluate_GNN()

    print("y_true_homo mean/std:", y_true_homo.mean(), y_true_homo.std())
    print("y_pred_homo mean/std:", y_pred_homo.mean(), y_pred_homo.std())
    print("y_true_homo min/max:", y_true_homo.min(), y_true_homo.max())
    print("y_pred_homo min/max:", y_pred_homo.min(), y_pred_homo.max())

    print("y_true_lumo mean/std:", y_true_lumo.mean(), y_true_lumo.std())
    print("y_pred_lumo mean/std:", y_pred_lumo.mean(), y_pred_lumo.std())
    print("y_true_lumo min/max:", y_true_lumo.min(), y_true_lumo.max())
    print("y_pred_lumo min/max:", y_pred_lumo.min(), y_pred_lumo.max())
    


#--------- direct gap influenced by HOMO/LUMO auxiliary tasks --------------#
    name_model_direct_gap_influenzed_HL = "direct_gap_influenzed_HL"

    datas = []

    for _, row in df.iterrows():
        smile = row["smiles"]

        # y[0] = gap
        # y[1] = homo
        # y[2] = lumo
        target = row["gap"], row["homo"], row["lumo"]

        data = graph.smiles_to_graph(smile, target)
        datas.append(data)

    train_data, test_data = graph.split_data_GNN(
        datas,
        test_size=0.2,
        random_state=42
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model_direct_gap_influenzed_HL = GNNModel(
        in_channels=8,
        hidden_channels=128,
        out_channels=3,
        positive_output=False
    )

    trainer = GNNTrainer(
        model_direct_gap_influenzed_HL,
        train_loader,
        test_loader,
        lr=0.001
    )

    print(trainer.device)
    trainer.run(epochs=10)

    trainer.save_model_gnn(
        model_direct_gap_influenzed_HL,
        name_model_direct_gap_influenzed_HL,
        output_path_model
    )

    y_true, y_pred = trainer.get_predictions()

    # prendiamo SOLO il gap
    y_true_gap = y_true[:, 0]
    y_pred_gap = y_pred[:, 0]

    visualizer.plot_predictions(
        y_true_gap,
        y_pred_gap,
        output_imm + "direct_gap_influenzed_HL.png",
        title="Direct gap influenced by HOMO/LUMO"
    )

    visualizer.plot_errors(
        y_true_gap,
        y_pred_gap,
        output_imm + "error_direct_gap_influenzed_HL.png",
        bins=20,
        title="Errors direct gap influenced by HOMO/LUMO"
    )

    trainer.evaluate_GNN()

    print("y_true_gap mean/std:", y_true_gap.mean(), y_true_gap.std())
    print("y_pred_gap mean/std:", y_pred_gap.mean(), y_pred_gap.std())
    print("y_true_gap min/max:", y_true_gap.min(), y_true_gap.max())
    print("y_pred_gap min/max:", y_pred_gap.min(), y_pred_gap.max())

#--------- end direct gap influenced by HOMO/LUMO auxiliary tasks --------------#


if __name__ == "__main__":
    build_model() 
