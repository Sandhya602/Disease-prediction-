# Disease Prediction Toolkit: Building and Evaluating ML Models

Author: **Sandhya Shree B V**

A beginner-friendly toolkit to train and evaluate ML models (Logistic Regression, Random Forest) for disease prediction using real-world health datasets.

## Features
- Robust preprocessing with `ColumnTransformer` (numeric imputation + scaling, categorical imputation + one-hot)
- Train Logistic Regression / Random Forest via CLI
- Metrics: accuracy, precision, recall, F1, ROC-AUC (binary), confusion matrix
- Plots: ROC curve, Confusion Matrix
- Saved artifacts: model, preprocessor, metrics JSON, plots

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
