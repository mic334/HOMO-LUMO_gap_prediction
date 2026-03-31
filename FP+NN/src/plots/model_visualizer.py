import matplotlib.pyplot as plt
import pandas as pd


class ModelVisualizer:
    def plot_predictions(self, y_true, y_pred, title="Predicted vs Actual"):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()]
        )
        plt.xlabel("Valori reali")
        plt.ylabel("Valori predetti")
        plt.title(title)
        plt.savefig("../figures/predicted_vs_actual.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_errors(self, y_true, y_pred, bins=30, title="Error Distribution"):
        errors = y_true - y_pred

        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=bins)
        plt.xlabel("Errore")
        plt.ylabel("Frequenza")
        plt.title(title)
        plt.savefig("../figures/error_distribution.png", dpi=300, bbox_inches="tight")  
        plt.show()
