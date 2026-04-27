# HOMO-LUMO Gap Prediction

Machine learning project for predicting molecular electronic properties using Graph Neural Networks (GNNs) built from SMILES representations.

---

## Overview

This project focuses on the prediction of key molecular electronic quantities from molecular graphs derived from SMILES strings.

At the current stage, the main targets are:

- **HOMO** energy
- **LUMO** energy
- **HOMO-LUMO gap**

These quantities are important in computational chemistry and are often related to:

- electronic structure
- chemical reactivity
- optical behavior
- molecular stability

The project is based on a graph-learning pipeline in which each molecule is converted from SMILES into a molecular graph and processed with a Graph Neural Network for regression.

---

## Current Branch

This branch (`graph_nn_benchmark`) contains the current experimental benchmark based on:

- QM9 raw data parsing
- SMILES extraction
- graph construction from molecules
- GNN training with PyTorch Geometric
- model evaluation and visualization
- prediction of:
  - HOMO
  - LUMO
  - HOMO-LUMO gap

---

## Current Pipeline

The current workflow is:

1. parse raw QM9 `.xyz` files
2. extract molecular information and target values
3. save the processed dataset as CSV
4. convert each SMILES string into a graph
5. build PyTorch Geometric `Data` objects
6. split the dataset into training and test sets
7. train a GNN regression model
8. save the trained model
9. generate evaluation plots for:
   - HOMO
   - LUMO
   - HOMO-LUMO gap

---

## Model

The current model is a Graph Neural Network implemented with PyTorch Geometric.

Main characteristics:

- node features extracted from RDKit atoms
- graph convolution layers (`GCNConv`)
- global pooling over node embeddings
- regression head for molecular property prediction

Depending on the experiment, the model can be used to predict:

- only the **HOMO-LUMO gap**
- both **HOMO** and **LUMO**

The current node-level features include:

- atomic number
- atom degree
- formal charge
- aromaticity
- total number of hydrogens
- implicit valence
- explicit valence
- ring membership

---

## Dataset

The current training pipeline uses the **QM9** molecular dataset in `.xyz` format.

Processed files are saved inside:

- `data/processed/`

An additional dataset of external molecules may also be used in future experiments for testing transferability and out-of-distribution behavior.

Alchemy dataset download:
- `https://alchemy.tencent.com`

---

## Project Structure

```text
HOMO-LUMO_gap_prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── PUB_processed/
├── figures/
├── models/
│   └── models.pth
├── notebooks/
├── results/
├── script/
│   ├── HOMO_LUMO.sh
│   ├── estrai_HOMO_LUMO_xtb.sh
│   └── sottometti.sh
├── src/
│   ├── pred.py
│   ├── result_modello.py
│   ├── graph/
│   │   └── graf.py
│   ├── modello/
│   │   └── modello_GNN.py
│   ├── parser/
│   │   └── data_loader.py
│   ├── plots/
│   │   └── model_visualizer.py
│   └── xtb_DFT/
├── requirements.txt
└── README.md
