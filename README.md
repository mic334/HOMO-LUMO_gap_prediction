# 🧪 HOMO-LUMO Gap Prediction with Machine Learning

A machine learning project for predicting the HOMO-LUMO energy gap of organic molecules using molecular descriptors derived from SMILES representations.

---

## 📌 Overview

The HOMO-LUMO gap is a key property in computational chemistry and materials science, related to:

* electronic structure
* optical properties
* reactivity

In this project, we build a complete pipeline that:

1. Parses raw QM9 dataset files
2. Extracts molecular features from SMILES
3. Trains a machine learning model
4. Evaluates performance and interprets results

---

## 📊 Dataset

* Source: QM9 dataset
* Contains:

  * molecular structures
  * quantum chemical properties
* Target:

  * **HOMO-LUMO gap**

Example data:

| molecule_id | smiles | homo    | lumo   | gap    |
| ----------- | ------ | ------- | ------ | ------ |
| 1           | C      | -0.3877 | 0.1171 | 0.5048 |

---

## ⚙️ Pipeline

### 1. Data Parsing

Raw `.xyz` files are parsed to extract:

* SMILES
* HOMO
* LUMO
* GAP

### 2. Feature Engineering

SMILES are converted into numerical descriptors using RDKit:

* Molecular weight
* LogP
* TPSA
* Number of rings
* Hydrogen bond donors/acceptors
* Rotatable bonds
* Valence electrons

### 3. Model Training

We use a:

👉 **Random Forest Regressor**

Steps:

* Train/test split (80/20)
* Model training
* Prediction on unseen data

---

## 📈 Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

These metrics quantify how well the model predicts the HOMO-LUMO gap.

---

## 📊 Results & Visualization

The project includes:

* Predicted vs Actual plot
* Error distribution histogram
* Feature importance analysis

### 🔍 Feature Importance

The Random Forest model highlights which molecular descriptors are most relevant for predicting the gap.

This helps interpret the model in a chemically meaningful way.

---

## 🧠 Key Insights

* Molecular size and structure strongly influence the HOMO-LUMO gap
* Structural descriptors provide useful predictive power
* Even simple ML models can capture meaningful chemical trends

---

## 📂 Project Structure

```
src/
├── parser/
│   └── data_loader.py
├── features/
│   └── feature_extractor.py
├── models/
│   └── model_trainer.py
├── visualization/
│   └── model_visualizer.py
└── main.py
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>
cd homo-lumo-gap-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python -m src.main
```

---

## 🛠️ Technologies Used

* Python
* RDKit
* scikit-learn
* pandas
* matplotlib

---

## 📌 Future Improvements

* Add molecular fingerprints (Morgan fingerprints)
* Hyperparameter tuning
* Deep learning models (GNNs)
* Extend to other quantum properties

---
