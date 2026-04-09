# HOMO-LUMO Gap Prediction

A machine learning project for predicting the **HOMO-LUMO gap** of molecules using **SMILES-derived molecular descriptors**, **RDKit**, and **scikit-learn**.

---

## Overview

The HOMO-LUMO gap is an important molecular property in computational chemistry, often related to:

- electronic structure
- chemical reactivity
- optical behavior
- molecular stability

This repository implements a complete end-to-end workflow starting from raw **QM9** molecular files and ending with model training, evaluation, and visualization.

The **`main` branch** contains the current stable baseline based on **RDKit molecular descriptors**.

A separate development branch is used for more experimental work, including **fingerprint-based features and model extensions**.

---

## Branches

### `main`
Stable baseline pipeline based on:

- QM9 raw data parsing
- SMILES processing
- RDKit descriptor extraction
- regression model training
- evaluation and visualization

### experimental branch
A separate branch is used for experiments beyond the baseline, such as:

- molecular fingerprints
- alternative feature spaces
- additional models and comparisons

This keeps the `main` branch cleaner and focused on the core project pipeline.

---

## Current Pipeline

The current `main` branch performs the following steps:

1. parse raw QM9 `.xyz` files
2. extract molecular information and target values
3. save a processed dataset as CSV
4. compute RDKit molecular descriptors from SMILES
5. train a regression model
6. evaluate model performance
7. generate result plots

---

## Features

At the moment, the `main` branch uses **molecular descriptors**, not fingerprints yet.

The extracted features include:

- molecular weight
- exact molecular weight
- LogP
- TPSA
- number of rings
- number of aromatic rings
- number of rotatable bonds
- number of H-bond donors
- number of H-bond acceptors
- number of heavy atoms
- number of valence electrons

These features provide a compact and interpretable representation of each molecule.

---

## Project Structure

```text
HOMO-LUMO_gap_prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── results/
├── src/
│   ├── main.py
│   ├── parser/
│   │   └── data_loader.py
│   ├── modello/
│   │   └── modello.py
│   ├── plots/
│   │   └── model_visualizer.py
│   └── xy/
│       └── FeatureExtractor.py
├── requirements.txt
└── README.md