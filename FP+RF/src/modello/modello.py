import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42, n_estimators=100,n_jobs=-1):
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
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

    def get_feature_importance(self, feature_names):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance_df
    
  