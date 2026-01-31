# 🌌 MALLORN Astronomical Classification Challenge

**Built with PriorLabs-TabPFN**

## Licensing
- **Original Code:** This repository is licensed under the Apache License 2.0.
- **Model:** This project uses **TabPFN2**, which is licensed under the [Prior Labs License v1.1](THIRD_PARTY_LICENSES/PRIOR_LABS.txt).
- [cite_start]**Attribution:** As per Section 10 of the Prior Labs License, any derivative AI models based on this work must include "TabPFN" at the beginning of the model name.

**Goal:** Identify rare Tidal Disruption Events (TDEs)—stars being torn apart by black holes—from photometric light curves.

**Competition:** [Kaggle Competition Link](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)

**Data Source:** Zwicky Transient Facility (ZTF) simulations for LSST.

## 🛠 Prerequisites

You must have **Conda** or **Miniconda** installed to handle the complex astronomy dependencies (`sncosmo`, `astropy`) required for this project.

### Hardware & Time Requirements

> [!WARNING]
> **Resource Intensive Steps**:
>
> 1.  **CPU Intensive**: The initial data processing `notebooks/data-process-mallorn.ipynb` runs on **CPU** and takes approximately **10 hours** to complete (on Kaggle free tier, as we've done). **For judging purposes, we recommend starting from `notebooks/further_data.ipynb` to save significant time**, or skip data preprocessing completely as the processed data is stable.
> 2.  **VRAM Intensive**: The deep learning tabular models (`notebooks/04_NODE.ipynb` and `notebooks/04_FT_Transformer.ipynb`) require at least **15GB VRAM** to train effectively.
> 3.  **Low Resource Friendly**: Other Neural Networks (MLP, ResNet) are currently optimized to run in **~30 minutes** on a machine with as little as **4GB VRAM**. They can be further tuned to utilize more resources for faster training if available.

Ensure your machine meets these requirements before attempting a full reproduction from raw data.

## � Installation & Setup

### 1. Environment Setup

Open your terminal in this directory and run:

```bash
# 1. Update/Create the environment from the file
conda env update --file environment.yml --prune

# 2. Activate the environment
conda activate MLFinal
```

### 2. Kaggle Authentication

Ensure your API key is in the default location so scripts can download data automatically:
* **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`
* **Linux/Mac:** `~/.kaggle/kaggle.json`

### 3. Data Download

Data download is handled in the exploration notebook:
1. Open `notebooks/01_exploration.ipynb`
2. Run the **"Setup"** cell at the top to download and unzip data to `data/raw`.

## 🏃 Execution Instructions

### Option A: Master Script (Recommended)

To run the entire pipeline from feature filtering to ensemble submission:

```bash
python run_all.py
```

This script handles dependencies and executes notebooks in the correct seeded order using `papermill`.

### Option B: Manual Execution (Fallback)

If the master script fails, running the notebooks manually in this exact order will replicate the pipeline. **Ensure you restart the kernel between notebooks to clear memory.**

#### Phase 1: Data Preparation
*(These steps produce the base features used by the models. If `data/processed` is already populated, you may skip to Phase 2.)*

1.  **`notebooks/data-process-mallorn.ipynb`** (Legacy)
    *   *Role:* Initial data cleaning and format conversion.
2.  **`notebooks/03_make_folds.ipynb`**
    *   *Role:* Generates `train_folds.csv` for 5-fold Stratified Validation.
3.  **`notebooks/further_data.ipynb`**
    *   *Role:* Advanced Feature Engineering (Bazin fits, PCA, Light-curve stats).
    *   *Output:* `further_train_features.parquet`, `further_test_features.parquet`
4.  **`notebooks/preprocessing_nn.ipynb`**
    *   *Role:* Normalization (RankGauss) specifically for Neural Networks.
    *   *Output:* `further_train_processed_nn.parquet`

#### Phase 2: Model Training
*(These notebooks use Optuna for hyperparameter tuning. They satisfy 100% deterministic reproducibility via seeded `TPESampler`.)*

5.  **`notebooks/04_XGBoost_v4.ipynb`**
    *   *Output:* `models/oof_xgb.csv`, `models/preds_xgb.csv`
6.  **`notebooks/04_LightGBM_v4.ipynb`**
    *   *Output:* `models/oof_lgb.csv`, `models/preds_lgb.csv`
7.  **`notebooks/04_CatBoost_v4.ipynb`**
    *   *Output:* `models/oof_cat.csv`, `models/preds_cat.csv`
8.  **`notebooks/04_NN_MLP.ipynb`** (Multilayer Perceptron)
    *   *Output:* `models/oof_mlp.csv`, `models/preds_mlp.csv`
9.  **`notebooks/04_ResNet_v4.ipynb`** (ResNet Architecture)
    *   *Output:* `models/oof_resnet.csv`, `models/preds_resnet.csv`

#### Phase 3: Ensembling

10. **`notebooks/05_ensemble.ipynb`**
11. **`notebooks/05_ensemble_hill.ipynb`**

## 📂 Project Structure

```text
MALLORN_Challenge/
│
├── data/
│   ├── raw/             # Original Kaggle CSVs (Do not edit)
│   └── processed/       # Cleaned parquet/csv files
│
├── models/              # Predictions and saved model artifacts
├── notebooks/           # All analysis and modelling notebooks
├── notebooks_executed/  # Output reports from run_all.py
├── src/                 # Helper scripts
│   └── config_loader.py
├── submissions/         # Final CSVs for leaderboard
├── config.yml           # Central configuration (Paths, Seeds, Constants)
├── environment.yml      # Conda environment definition
├── run_all.py           # Master execution pipeline
└── README.md            # Project documentation
```

## 🧠 Models & Architecture

We employ a diverse ensemble of 8 distinct models, combining state-of-the-art Gradient Boosting Decision Trees (GBDT) with modern Deep Learning architectures for tabular data.

### Tree-Based Models
1.  **XGBoost (v4)**: Optimized for speed and performance with gradient boosting.
2.  **LightGBM (v4)**: Uses leaf-wise tree growth, faster training speed and higher efficiency.
3.  **CatBoost (v4)**: Handles categorical features automatically and reduces overfitting with ordered boosting.
4.  **ExtraTrees**: An extremely randomized tree ensemble, reducing variance compared to standard Random Forests.

### Deep Learning Models
5.  **Multilayer Perceptron (MLP)**: A classic fully connected deep neural network with RankGauss normalization.
6.  **ResNet**: Adapted Residual Network architecture for tabular data, allowing for deeper networks without vanishing gradients.
7.  **NODE (Neural Oblivious Decision Ensembles)**: A deep learning architecture designed to mimic the oblivious decision trees of GBDT, differentiable and end-to-end trainable.
8.  **FT-Transformer**: Feature Tokenizer + Transformer, applying attention mechanisms to tabular features, often matching or beating GBDT performance.

## ⚙️ Optimization Strategy

Our optimization pipeline ensures maximum performance while maintaining strict reproducibility:

*   **Optuna & TPE**: Hyperparameters are tuned using Optuna with the Tree-structured Parzen Estimator (TPE) sampler.
*   **Deterministic Tuning**: We enforce fixed random seeds for the TPE sampler to ensure that the "random" search is identical every run.
*   **Stratified Cross-Validation**: All tuning uses 5-Fold Stratified CV to prevent overfitting to a specific data subset.
*   **Metric**: The primary optimization metric is **Average Precision (PR-AUC)** (or F1 Score for final thresholding) to handle the class imbalance effectively.

## 🤝 Ensembling Strategy

To maximize generalization, we combine our models using two strategies:

1.  **Weighted Averaging**: Basic ensemble combining predictions based on validation scores.
2.  **Hill Climbing Optimization**: (`notebooks/05_ensemble_hill.ipynb`) An iterative algorithm that finds the optimal weights for a weighted average by greedily adding models that improve the ensemble score.
3.  **Grandmaster Ensemble Strategy**: We utilize varying "views" of the data (e.g., standard features vs. strictly selected features) across different models to uncorrelated errors and boost ensemble performance.