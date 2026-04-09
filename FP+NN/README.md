# FP+NN

This subproject explores **neural-network-based prediction** of the HOMO-LUMO gap using **Morgan fingerprint** features.

## Purpose

`FP+NN` is the most flexible and most experimental track in the fingerprint benchmark branch.

Unlike the tree-based folders, this one appears to be used not only for benchmarking but also for a more operational workflow, with dedicated `models/` and `script/` directories.

## Model

This experiment uses:

- **Morgan fingerprints** as molecular representation
- **MLP / neural-network-style regression** as the main modeling direction

Neural models are included here to test whether a more flexible function approximator can learn useful patterns from fingerprint vectors beyond what tree ensembles capture.

## Why this folder exists

This folder isolates the neural-network track from the more classical ensemble approaches.

Compared with the other tracks:

- it is less plug-and-play than `FP+RF`
- it usually requires more tuning and preprocessing
- it is the most likely place for extended experimentation and workflow scripts

## Folder Contents

This subproject includes:

- `src/` for core code
- `models/` for saved or reusable model artifacts
- `script/` for shell workflows and execution utilities
- `data/`, `figures/`, `notebooks/`, and `results/`
- `requirements.txt` for dependencies

## Practical Role

This is the most “research-like” folder in the branch.

It is the right place for:

- trying training variants
- saving model artifacts
- running repeatable scripts
- testing prediction-oriented workflows

## Notes

- neural models can be more expressive, but they are usually less stable than tree-based baselines
- this folder appears broader than a simple benchmark, so it may evolve into a more applied experiment area
- model behavior should always be compared back to `FP+RF` and `FP+HGB`
