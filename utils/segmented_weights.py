import numpy as np
import optuna
from sklearn.metrics import balanced_accuracy_score

def optimize_segmented_weights(oof_probs, y_true, segment_mask,optimizer):
    segment_weights = {}
    for name, mask in segment_mask.items():
        weights, score = optimizer(oof_probs[mask], y_true[mask])

        segment_weights[name] = weights

        print(
            f"{name}: "
            f"n={mask.sum()}, "
            f"score={score:.5f}, "
            f"weights={np.round(weights, 4)}"
        )

    return segment_weights

def apply_segmented_weights(
    probs,
    segment_masks,
    segment_weights
):
    preds = np.empty(
        len(probs),
        dtype=int
    )

    for name, mask in segment_masks.items():

        preds[mask] = (
            probs[mask]
            * segment_weights[name]
        ).argmax(axis=1)

    return preds