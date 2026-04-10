import pandas as pd
import numpy as np
import time
import os

#per GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

  
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        # Convoluzione grafo: feature iniziali -> feature nascoste
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Seconda convoluzione grafo
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Output finale per regressione
        self.lin1 = Linear(hidden_channels,hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # Primo messaggio tra nodi
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Secondo messaggio tra nodi
        x = self.conv2(x, edge_index)
        #ReLU significa Rectified Linear Unit ed è una funzione di attivazione. In PyTorch viene definita elemento per elemento come max(0, x). 
        #Quindi lascia passare i valori positivi e mette a zero quelli negativi
        x = F.relu(x)

        # Aggregazione da nodi -> grafo
        x = global_add_pool(x, batch)

        # Predizione finale
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x

class GNNTrainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001, device=None):
        """
        model: modello GNN già creato
        train_loader: DataLoader per il training
        test_loader: DataLoader per il test
        lr: learning rate dell'optimizer
        device: cpu o cuda
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.test_losses = []
        self.epoch_time = []
        self.y_true = None
        self.y_pred = None
        
        # Se non viene specificato, usa GPU se disponibile, altrimenti CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sposta il modello sul device scelto
        self.model.to(self.device)

        # Loss per regressione
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self):
        """
        Esegue una singola epoca di training.
        Restituisce la loss media sul training set.
        """

        # modalità training
        self.model.train()

        total_loss = 0

        for batch in self.train_loader:
            # sposta il batch su device
            batch = batch.to(self.device)

            # azzera i gradienti del passo precedente
            self.optimizer.zero_grad()

            # forward pass
            pred = self.model(batch.x, batch.edge_index, batch.batch)

            # target reale
            target = batch.y.view(-1, 1).float()

            # calcolo loss
            loss = self.criterion(pred, target)

            # backward
            loss.backward()

            # aggiornamento pesi
            self.optimizer.step()

            total_loss += loss.item()

        # loss media sull'intera epoca
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self):
        """
        Valuta il modello sul test set.
        Restituisce la loss media di test.
        """

        # modalità evaluation
        self.model.eval()

        total_loss = 0

        # disattiva il calcolo dei gradienti
        with torch.no_grad():
            for batch in self.test_loader:
                # sposta batch su device
                batch = batch.to(self.device)

                # forward pass
                pred = self.model(batch.x, batch.edge_index, batch.batch)
                

                # target reale
                target = batch.y.view(-1, 1).float()

                # loss
                loss = self.criterion(pred, target)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        return avg_loss

    def run(self, epochs=20):
        """
        Esegue training + evaluation per un certo numero di epoche.
        """
        total_start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            train_loss = self.train_one_epoch()
            test_loss = self.evaluate()
            epoch_time = time.time() - epoch_start_time

            #crea liste !sentinella!
            self.epoch_time.append(epoch_time)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} |" 
                f"Time: {epoch_time:.2f} s"
            )
        total_time = time.time() - total_start_time
        print(f"Training completato in {total_time:.2f} secondi")
        
    def get_predictions(self):
        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)

                pred = self.model(batch.x, batch.edge_index, batch.batch)

                y_true.append(batch.y.view(-1, 1).cpu())
                y_pred.append(pred.cpu())

        y_true = torch.cat(y_true, dim=0).numpy().flatten()
        y_pred = torch.cat(y_pred, dim=0).numpy().flatten()
        self.y_true = y_true
        self.y_pred = y_pred 

        return y_true, y_pred
    
    def evaluate_GNN(self):
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        r2 = r2_score(self.y_true, self.y_pred)

        self.metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print("Valutazione modello:")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")

        return self.metrics
    
    def save_model_gnn(self,model,model_name,model_path=None):
        if model_path is None:
            model_path = "."

        os.makedirs(model_path, exist_ok=True)
        percorso_file = f"{model_path}/{model_name}.pth"

        torch.save(model, percorso_file)
        print(f"Modello salvato in: {percorso_file}")

        return