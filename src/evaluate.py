import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

from utils import ensure_dirs, save_json

def evaluate(csv: str, target: str, model_path: str, preprocessor_path: str, output_dir: str):
    models_dir, reports_dir = ensure_dirs(output_dir)

    df = pd.read_csv(csv)
    assert target in df.columns, f"Target column '{target}' not found."

    if df[target].dtype == object:
        df[target] = df[target].astype(str).str.strip().str.lower().map({"yes":1, "no":0, "1":1, "0":0}).fillna(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    y_pred = pipe.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="binary" if y.nunique()==2 else "weighted", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="binary" if y.nunique()==2 else "weighted", zero_division=0)),
        "f1": float(f1_score(y, y_pred, average="binary" if y.nunique()==2 else "weighted", zero_division=0)),
    }

    # ROC-AUC binary only
    if y.nunique() == 2 and hasattr(model, "predict_proba"):
        y_proba = pipe.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
        RocCurveDisplay.from_predictions(y, y_proba)
        plt.title("ROC Curve (Evaluation)")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "eval_roc.png"), dpi=200)
        plt.close()

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format="d")
    plt.title("Confusion Matrix (Evaluation)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "eval_confusion_matrix.png"), dpi=200)
    plt.close()

    save_json(metrics, os.path.join(reports_dir, "evaluation_metrics.json"))
    print("Saved evaluation metrics and plots to:", reports_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a dataset")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--model_path", required=True, help="Path to saved model .joblib")
    parser.add_argument("--preprocessor_path", required=True, help="Path to saved preprocessor .joblib")
    parser.add_argument("--output_dir", default=".", help="Base output directory")
    args = parser.parse_args()
    evaluate(args.csv, args.target, args.model_path, args.preprocessor_path, args.output_dir)
  
