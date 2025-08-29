Disease Prediction Toolkit: Building and
Evaluating ML Models
Author: Sandhya Shree B V
A beginner-friendly toolkit to train and evaluate ML models (Logistic Regression, Random Forest)
for disease prediction using real-world health datasets. The project demonstrates a complete
machine learning workflow from preprocessing to evaluation and predictions.
Features
• Robust preprocessing with ColumnTransformer (numeric imputation + scaling, categorical
imputation + one-hot).
• Train Logistic Regression / Random Forest via CLI.
• Metrics: accuracy, precision, recall, F1, ROC-AUC (binary), confusion matrix.
• Plots: ROC curve, Confusion Matrix.
• Saved artifacts: model, preprocessor, metrics JSON, plots.
Installation
1. Create a virtual environment:
python -m venv .venv
2. Activate it:
- Linux/Mac: source .venv/bin/activate
- Windows: .venv\Scripts\activate
3. Install requirements:
pip install -r requirements.txt
Usage
Train Logistic Regression:
python src/train_logreg.py --csv data/health.csv --target disease --output_dir . --test_size 0.2
--random_state 42
Train Random Forest:
python src/train_random_forest.py --csv data/health.csv --target disease --output_dir . --test_size
0.2 --random_state 42
Evaluate a Model:
python src/evaluate.py --csv data/health.csv --target disease --model_path
models/logreg_model.joblib --preprocessor_path models/preprocessor.joblib --output_dir .
Run Predictions:
python src/predict.py --csv data/new_patients.csv --model_path models/logreg_model.joblib
--preprocessor_path models/preprocessor.joblib --output_path predictions.csv# Disease-prediction-
