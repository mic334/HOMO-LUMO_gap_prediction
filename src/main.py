import pandas as pd
from parser.data_loader import QM9Parser
from xy.FeatureExtractor import FeatureExtractor
from modello.modello import  ModelTrainer
from plot.model_visualizer import ModelVisualizer

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

    # obj 
    extractor = FeatureExtractor()
    
    features_df, failed = extractor.transform(df)

    print(f"Feature dataset: {len(features_df)} righe")

    # STEP 4 — X e y
    X, y = extractor.get_xy(features_df)

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    #obj
    trainer = ModelTrainer()
    #train pipeline (X,y) -> split, train, predict, evaluate
    metrics = trainer.train_pipeline(X, y)
    print("Metriche finali:", metrics)
    #get feature importance (extractor.feature_names ) -> print importance  
    importance_df = trainer.get_feature_importance(extractor.feature_names)
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

    importance_df = trainer.get_feature_importance(X.columns)

    visualizer.plot_feature_importance(
        importance_df,
        top_n=10,
        title="Random Forest: Top 10 Feature Importances"
    )
    
    
    print("Pipeline completata!")

if __name__ == "__main__":
    main()