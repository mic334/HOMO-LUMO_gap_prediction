from xml.parsers.expat import errors
import matplotlib.pyplot as plt
import pandas as pd


class ModelVisualizer:
    def plot_predictions(self, y_true, y_pred, output_imm ,title="Predicted vs Actual"):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()]
        )
        plt.xlabel("Valori reali")
        plt.ylabel("Valori predetti")
        plt.title(title)
        plt.savefig(output_imm, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_errors(self, y_true, y_pred, output_imm , bins=30, title="Error Distribution"):
        errors = y_true - y_pred

        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=bins)
        plt.xlabel("Errore")
        plt.ylabel("Frequenza")
        plt.title(title)
        plt.savefig(output_imm, dpi=300, bbox_inches="tight")  
        plt.show()

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n=10, title="Feature Importance"):
        plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

        plt.figure(figsize=(8, 5))
        plt.barh(plot_df["feature"], plot_df["importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(title)
        plt.savefig("../results_models/feature_importance.png", dpi=300, bbox_inches="tight")  
        plt.show()
        
    def plot_losses(self, train_losses, test_losses, title="Training and Test Loss"):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epoche")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.savefig("../results_models/loss_curve.png", dpi=300, bbox_inches="tight")
        plt.show()


    def plot_epoch_times(self, epoch_time, title="Time per Epoch"):
        epochs = range(1, len(epoch_time) + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, epoch_time)
        plt.xlabel("Epoche")
        plt.ylabel("Tempo (s)")
        plt.title(title)
        plt.savefig("../results_models/epoch_times.png", dpi=300, bbox_inches="tight")
        plt.show()