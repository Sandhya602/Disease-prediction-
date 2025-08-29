from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df.drop(columns=[target_col])
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
    categorical_pipeline = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", __import__("sklearn").pipeline.Pipeline(numeric_pipeline), numeric_features),
            ("cat", __import__("sklearn").pipeline.Pipeline(categorical_pipeline), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features
  
