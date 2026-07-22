import pandas as pd
import numpy as np

def predict_probe(df, probe):
    
    model = probe["model"]
    features = probe["features"]
    categorical_features = probe["categorical_features"]

    X = df[features].copy()

    for col in categorical_features:
        X[col] = X[col].astype("category")

    return model.predict_proba(X)