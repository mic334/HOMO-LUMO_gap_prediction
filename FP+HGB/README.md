## 🧪 Fingerprint Benchmark Branch

This branch extends the baseline project by introducing **molecular fingerprints (Morgan/ECFP)** as an alternative feature representation for predicting the HOMO-LUMO gap.

### 🔍 What’s new

* Added **Morgan fingerprint extraction (2048 bits)**
* Replaced descriptor-based features with **fingerprint-based features**
* Enabled comparison between:

  * Descriptors (baseline)
  * Fingerprints
  * Hybrid (descriptors + fingerprints)

### ⚙️ Model

* HistGradientBoostingRegressor

### 🎯 Goal

Evaluate whether molecular fingerprints improve predictive performance over classical descriptors.

### 📊 Notes

* Fingerprints provide better representation power but lower interpretability
* Designed for experimentation and benchmarking

