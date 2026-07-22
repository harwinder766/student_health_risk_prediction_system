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


def train_probe(train,target,features,categorical_cols,missing_mask,
                leak_features=None,n_splits=5,seed=42,
                class_weight="balanced",verbose=True):
    """
    Train a LightGBM probe model using only rows where the probe
    target is available.

    Returns
    -------
    probe_models : list
        One trained model per CV fold.

    oof_probs : np.ndarray
        OOF probabilities for rows where target is available.

    label_encoder : LabelEncoder
        Encoder fitted on the probe target.

    probe_features : list
        Features actually used by the probe.

    probe_cats : list
        Categorical features used by the probe.

    score : float
        OOF balanced accuracy.
    """

    if leak_features is None:
        leak_features = []

    # Rows where target is available
    probe = train.loc[~missing_mask].reset_index(drop=True)

    # Encode probe target
    label_encoder = LabelEncoder()
    y_probe = label_encoder.fit_transform(probe[target])

    # Remove target/leakage features
    probe_features = [
        f for f in features
        if f not in leak_features and f != target
    ]

    probe_cats = [
        c for c in categorical_cols
        if c in probe_features
    ]

    X_probe = probe[probe_features].copy()

    for col in probe_cats:
        X_probe[col] = X_probe[col].astype("category")

    n_classes = len(label_encoder.classes_)

    oof_probs = np.zeros(
        (len(probe), n_classes)
    )

    probe_models = []

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold, (tr_idx, val_idx) in enumerate(
        cv.split(X_probe, y_probe),
        1
    ):

        model = LGBMClassifier(objective="multiclass",num_class=n_classes,n_estimators=500,
                               learning_rate=0.1,class_weight=class_weight,
                               random_state=seed,verbosity=-1)

        model.fit(
            X_probe.iloc[tr_idx],
            y_probe[tr_idx]
        )

        oof_probs[val_idx] = model.predict_proba(
            X_probe.iloc[val_idx]
        )

        probe_models.append(model)

        if verbose:
            print(f"Fold {fold} done")

    # OOF evaluation
    oof_pred = oof_probs.argmax(axis=1)

    acc = accuracy_score(
        y_probe,
        oof_pred
    )

    b_acc = balanced_accuracy_score(
        y_probe,
        oof_pred
    )

    if verbose:

        majority_baseline = (
            pd.Series(y_probe)
            .value_counts(normalize=True)
            .max()
        )

        print(f"\nProbe rows: {len(probe):,}")
        print(f"Features: {len(probe_features)}")

        print(
            f"\nAccuracy          : {acc:.4f}"
            f" (majority baseline {majority_baseline:.4f})"
        )

        print(
            f"Balanced Accuracy : {b_acc:.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                y_probe,
                oof_pred,
                target_names=label_encoder.classes_.astype(str),
                digits=4
            )
        )

    return {
        "models": probe_models,
        "oof_probs": oof_probs,
        "label_encoder": label_encoder,
        "features": probe_features,
        "categorical_features": probe_cats,
        "balanced_accuracy": b_acc
    }