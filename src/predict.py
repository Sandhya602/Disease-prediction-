import argparse
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

def predict(csv: str, model_path: str, preprocessor_path: str, output_path: str):
    X = pd.read_csv(csv)

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # If CSV accidentally contains the target, drop it
    for t in ["target", "label", "y", "disease"]:
        if t in X.columns:
            X = X.drop(columns=[t])

    preds = pipe.predict(X)
    out = X.copy()
    out["prediction"] = preds

    # Add probability if available (binary classification)
    if hasattr(model, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba.shape[1] == 2:
            out["probability_positive"] = proba[:, 1]

    out.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new data")
    parser.add_argument("--csv", required=True, help="Path to input CSV (no labels required)")
    parser.add_argument("--model_path", required=True, help="Path to saved model .joblib")
    parser.add_argument("--preprocessor_path", required=True, help="Path to saved preprocessor .joblib")
    parser.add_argument("--output_path", required=True, help="Where to save predictions CSV")
    args = parser.parse_args()
    predict(args.csv, args.model_path, args.preprocessor_path, args.output_path)
  
