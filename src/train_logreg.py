import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
)

from utils import ensure_dirs, save_json
from preprocess import build_preprocessor

def train_logreg(csv: str, target: str, output_dir: str, test_size: float, random_state: int):
    models_dir, reports_dir = ensure_dirs(output_dir)

    df = pd.read_csv(csv)
    assert target in df.columns, f"Target column '{target}' not found."

    # Basic target cleaning (optional): convert 'yes'/'no' -> 1/0 if present
    if df[target].dtype == object:
        df[target] = df[target].astype(str).str.strip().str.lower().map({"yes":1, "no":0, "1":1, "0":0}).fillna(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    preprocessor, _, _ = build_preprocessor(df, target)

    model = LogisticRegression(max_iter=1000, n_jobs=None)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None)

    pipe.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(pipe.named_steps["preprocessor"], os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(pipe.named_steps["model"], os.path.join(models_dir, "logreg_model.joblib"))

    # Eval
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary" if y.nunique()==2 else "weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="binary" if y.nunique()==2 else "weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="binary" if y.nunique()==2 else "weighted", zero_division=0)),
    }

    # ROC-AUC (binary only)
    if y.nunique() == 2:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        # ROC curve
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("Logistic Regression ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "logreg_roc.png"), dpi=200)
        plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "logreg_confusion_matrix.png"), dpi=200)
    plt.close()

    save_json(metrics, os.path.join(reports_dir, "logreg_metrics.json"))
    print("Training complete.")
    print("Saved:", os.path.join(models_dir, "logreg_model.joblib"))
    print("Metrics JSON:", os.path.join(reports_dir, "logreg_metrics.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression for disease prediction")
    parser.add_argument("--csv", required=True, help="Path to input dataset CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output_dir", default=".", help="Base output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train_logreg(args.csv, args.target, args.output_dir, args.test_size, args.random_state)

