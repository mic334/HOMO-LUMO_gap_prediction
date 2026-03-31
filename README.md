## 🧪 Model Comparison Branch

This branch explores different machine learning approaches for predicting the HOMO-LUMO gap using molecular fingerprints.

### 📁 Structure

* **FP + Random Forest**

  * Baseline model using Morgan fingerprints
  * Robust and strong performance on tabular data

* **FP + HistGradientBoosting**

  * Boosting-based model
  * Faster and often competitive or better than Random Forest on larger datasets

* **FP + MLP (Neural Network)**

  * Feedforward neural network using fingerprint features
  * Requires feature scaling and tuning
  * More flexible but less stable than tree-based models

### 🎯 Goal

Compare different models on the same feature representation (Morgan fingerprints) to evaluate performance differences.

### 📊 Notes

* Tree-based models are generally more stable on structured data
* Neural networks may require tuning but can capture complex patterns
* Experiments are performed on reduced datasets for faster iteration, with full QM9 used for final evaluation

