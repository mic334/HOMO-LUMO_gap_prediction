import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    def __init__(self, test_size=0.2, 
                 random_state=42, 
                 hidden_layer_sizes=(100,), 
                 activation='relu', 
                 solver='adam', 
                 max_iter=300, 
                 learning_rate_init=0.001, 
                 batch_size=256,early_stopping=True,
                 n_iter_no_change=10,
                 alpha=0.0001):
        self.test_size = test_size
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.alpha = alpha
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
           hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=self.alpha,          # regolarizzazione
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,   # 🔥 importantissimo
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state
)
        


        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = None

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        print("Train shape:", self.X_train.shape)
        print("Test shape :", self.X_test.shape)

    def train(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.model.fit(self.X_train, self.y_train)
        print("Training completato.")

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)

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

    def train_pipeline(self, X, y):
        self.split_data(X, y)
        self.train()
        self.predict()
        return self.evaluate()


    
  