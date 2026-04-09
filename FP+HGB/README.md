# FP+HGB

This subproject benchmarks **HistGradientBoosting regression** on **Morgan fingerprint** features for HOMO-LUMO gap prediction.

## Purpose

`FP+HGB` explores a boosting-based alternative to Random Forest on the same fingerprint representation.

The idea is to test whether a more aggressive ensemble method can extract better predictive signal from high-dimensional binary molecular features.

## Model

This experiment uses:

- **Morgan fingerprints** as molecular representation
- **HistGradientBoostingRegressor** as prediction model

This setup is interesting because boosting models are often:

- faster than large random forests
- competitive on structured datasets
- better at capturing sharper nonlinear patterns

## Why this folder exists

This folder isolates the fingerprint + boosting setup from the other experiments in the branch.

Compared with the other tracks:

- it is usually more performance-oriented than `FP+RF`
- it is less workflow-heavy than `FP+NN`
- it acts as the “boosting candidate” in the benchmark branch

## Folder Contents

This subproject includes:

- `src/` for the main code pipeline
- `data/` for dataset artifacts
- `figures/` and `results/` for outputs
- `requirements.txt` for local dependencies

## Interpretation

This is the experiment meant to test whether boosting can outperform the more standard Random Forest baseline on fingerprint inputs.

If `FP+HGB` wins consistently, it becomes a strong candidate for the best tree-based model in this branch.

## Notes

- fingerprint representations are richer than classical descriptors but harder to interpret
- this folder is experimental and benchmark-oriented
- final conclusions should be based on direct comparison with `FP+RF` and `FP+NN`
