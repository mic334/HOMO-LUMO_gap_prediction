import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MathBase:
    def mae(self, y_true, y_pred):
    
        return mean_absolute_error(y_true, y_pred)

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
    
    
    def all_metrics(self, y_true, y_pred):
        mae = self.mae(y_true, y_pred)
        rmse = self.rmse(y_true, y_pred)
        r2 = self.r2(y_true, y_pred)
        print("inizio metriche")
        print(f"mae : {mae}" ," più basso = meglio ")
        print(f"rmese : {rmse}",  " più basso = meglio ")
        print(f"R quadro : {r2}", " più alto = meglio ")
        print("fine metriche")
        return mae, rmse, r2