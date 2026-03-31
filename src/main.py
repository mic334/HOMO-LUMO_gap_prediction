import pandas as pd
from parser.data_loader import QM9Parser
from xy.FeatureExtractor import FeatureExtractor
from modello.modello import ModelTrainer
from plots.model_visualizer import ModelVisualizer
from rdkit import Chem

def main():
    input_path = "../data/raw/qm9_dataset"
    output_path = "../data/processed/qm9_gap_dataset.csv"

    parser = QM9Parser(input_path)
    print(parser.folder_path)

    df = parser.parse_folder()
    parser.save_csv(df, output_path)

    df = pd.read_csv(output_path)
    print(f"Dataset caricato: {len(df)} righe")

    extractor = FeatureExtractor()

    features_df, failed = extractor.transform(df)
    print(f"Feature dataset: {len(features_df)} righe")

    # SMILES -> molecules
    features_df["molecules"] = [Chem.MolFromSmiles(smiles) for smiles in features_df["smiles"]]
    features_df = features_df[features_df["molecules"].notnull()].reset_index(drop=True)

    # target
    y = features_df["gap"].values
    print("Shape y:", y.shape)

    # Morgan fingerprints
    X = extractor.compute_morgan_fingerprints(features_df["molecules"])
    print("Shape X (fingerprints):", X.shape)

    trainer = ModelTrainer()
    metrics = trainer.train_pipeline(X, y)
    print("Metriche finali:", metrics)

    fp_feature_names = [f"fp_{i}" for i in range(X.shape[1])]
    importance_df = trainer.get_feature_importance(fp_feature_names)

    print("Importanza delle feature:")
    print(importance_df)

    visualizer = ModelVisualizer()

    visualizer.plot_predictions(
        trainer.y_test,
        trainer.y_pred,
        title="Random Forest: Predicted vs Actual"
    )

    visualizer.plot_errors(
        trainer.y_test,
        trainer.y_pred,
        bins=30,
        title="Random Forest: Error Distribution"
    )

    visualizer.plot_feature_importance(
        importance_df,
        top_n=10,
        title="Random Forest: Top 10 Fingerprint Importances"
    )

    print("Pipeline completata!")

if __name__ == "__main__":
    main()