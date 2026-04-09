# Fingerprint Benchmark Branch

This branch is dedicated to benchmarking different machine learning models for **HOMO-LUMO gap prediction** using **molecular fingerprints** instead of the descriptor-only baseline available in `main`.

## Purpose

The goal of this branch is to explore whether **Morgan fingerprints (ECFP-like representations)** provide a stronger feature space for regression compared with the descriptor-based pipeline used in the main branch.

Rather than keeping all experiments mixed together, this branch separates them into model-specific subprojects.

## Branch Structure

This branch currently contains three benchmark tracks:

- **FP+RF** — fingerprint features + Random Forest
- **FP+HGB** — fingerprint features + HistGradientBoosting
- **FP+NN** — fingerprint features + neural network / MLP-based approach

Each folder is organized as an independent experiment space and may include its own:

- source code
- notebooks
- data artifacts
- figures
- results
- local README

## Why a Separate Branch

The `main` branch is kept focused on the cleaner and more stable **descriptor-based baseline**.

This branch is used for:

- model comparison
- fingerprint-based feature engineering
- experimental benchmarking
- testing alternatives without cluttering the main pipeline

## Feature Representation

The experiments in this branch are based on **Morgan fingerprints**, used as a higher-dimensional molecular representation than standard RDKit descriptors.

This makes the branch more suitable for benchmarking predictive performance, even if the resulting models are generally less interpretable than the descriptor-based baseline.

## Available Experiments

### FP+RF
Random Forest baseline on fingerprint features.

### FP+HGB
Gradient boosting approach on the same fingerprint representation.

### FP+NN
Neural network experiment using fingerprint vectors as input.

## Notes

- This branch is experimental by design.
- Performance, preprocessing, and implementation details may differ across subfolders.
- The detailed documentation for each setup should live inside the corresponding experiment directory.

## Suggested Navigation

If you are looking for the stable baseline pipeline, check the `main` branch.

If you want model-specific details, go directly into:

- `FP+RF/`
- `FP+HGB/`
- `FP+NN/`

## Project Direction

This branch is intended as a comparison layer on top of the main project:

- `main` = descriptor-based baseline
- `finger_prints_benchmark` = fingerprint-based benchmark branch

It is meant to help evaluate trade-offs between:

- interpretability
- feature richness
- model complexity
- predictive performance