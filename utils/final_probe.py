import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    classification_report
)

from lightgbm import LGBMClassifier


def train_final_probe(
    train,
    target,
    features,
    categorical_cols,
    missing_mask,
    leak_features=None,
    seed=42,
    class_weight="balanced"
):
    if leak_features is None:
        leak_features = []

    probe = train.loc[~missing_mask].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(probe[target])

    probe_features = [
        f for f in features
        if f not in leak_features and f != target
    ]

    probe_cats = [
        c for c in categorical_cols
        if c in probe_features
    ]

    X = probe[probe_features].copy()

    for col in probe_cats:
        X[col] = X[col].astype("category")

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        n_estimators=500,
        learning_rate=0.1,
        class_weight=class_weight,
        random_state=seed,
        verbosity=-1
    )

    model.fit(X, y)

    return {
        "model": model,
        "label_encoder": label_encoder,
        "features": probe_features,
        "categorical_features": probe_cats,
    }