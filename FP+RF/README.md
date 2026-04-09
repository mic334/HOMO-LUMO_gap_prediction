# FP+RF

This subproject benchmarks **Random Forest regression** on **Morgan fingerprint** features for HOMO-LUMO gap prediction.

## Purpose

`FP+RF` is the most natural tree-based baseline in the fingerprint branch.

It is meant to answer a simple question:

**how far can a robust, low-maintenance ensemble model go when molecular descriptors are replaced by fingerprint vectors?**

## Model

This experiment uses:

- **Morgan fingerprints** as molecular representation
- **Random Forest Regressor** as prediction model

Random Forest is a good benchmark here because it is:

- stable
- easy to train
- strong on tabular representations
- less sensitive than neural models to tuning

## Why this folder exists

This folder isolates the fingerprint + Random Forest setup from the other experiments in the branch.

Compared with the other tracks:

- it is usually easier to train than `FP+NN`
- it is more conservative than `FP+HGB`
- it works well as a reference baseline for fingerprint-based modeling

## Folder Contents

This subproject includes:

- `src/` for the main code pipeline
- `data/` for input or processed artifacts
- `notebooks/` for exploratory work
- `figures/` and `results/` for outputs
- `requirements.txt` for local dependencies

## Interpretation

This is the “strong baseline” experiment of the fingerprint branch.

If a more complex model does not clearly outperform this setup, then the added complexity may not be worth it.

## Notes

- fingerprint features improve structural representation but reduce interpretability
- this folder is intended for benchmarking, not as the final polished pipeline
- detailed metrics and outputs should be documented in `results/` or notebooks
