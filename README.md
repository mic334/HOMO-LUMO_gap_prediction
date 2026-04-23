# HOMO-LUMO Gap Prediction

Machine learning project for predicting the HOMO-LUMO gap of molecules using graph neural networks (GNNs) built from SMILES representations.

---

## Overview

The HOMO-LUMO gap is an important molecular property in computational chemistry and is often related to:

- electronic structure
- chemical reactivity
- optical behavior
- molecular stability

This branch focuses on a graph-based pipeline where each molecule is converted from SMILES to a molecular graph and processed with a Graph Neural Network for regression.

In addition to training on QM9-derived data, the project also includes scripts for running inference on external molecular datasets and comparing model predictions with xTB/DFT calculations.

---

## Current Branch

This branch (`graph_nn_benchmark`) contains the current experimental benchmark based on:

- QM9 raw data parsing
- SMILES extraction
- graph construction from molecules
- GNN training with PyTorch Geometric
- model evaluation and visualization
- prediction on external compounds
- comparison against xTB / DFT reference values

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
9. generate evaluation plots
10. run prediction on external molecules
11. compare predictions with xTB / DFT results

---

## Model

The current model is a graph neural network implemented with PyTorch Geometric.

Main characteristics:

- node features extracted from RDKit atoms
- graph convolution layers (`GCNConv`)
- global pooling over node embeddings
- final regression head for HOMO-LUMO gap prediction

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

The training pipeline uses QM9 molecular files in `.xyz` format.

**Dataset source:  https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/ name data = Compound_000000001_000500000.sdf.gz **

Processed files are saved inside:

- `data/processed/`
- `data/PUB_processed/` for external prediction datasets

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
│   ├── main.py
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

# alchemy dataset download
https://alchemy.tencent.com