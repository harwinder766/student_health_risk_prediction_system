import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_oof(
    y_true,
    probs,
    class_names,
    class_weights=None,
    title="OOF"
):
    """
    Evaluate multiclass OOF probabilities.

    Parameters
    ----------
    y_true : array-like
        True encoded class labels.

    probs : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).

    class_names : array-like
        Original class names in the same order as probability columns.

    class_weights : array-like, optional
        Decision-rule weights applied to probability columns.
        If None, normal argmax is used.

    title : str
        Name displayed in plots.

    Returns
    -------
    dict
        Balanced accuracy, predictions, confusion matrix,
        percentage confusion matrix, and weighted probabilities.
    """

    # -------------------------
    # Apply decision weights
    # -------------------------

    if class_weights is not None:
        final_probs = probs * np.asarray(class_weights)
    else:
        final_probs = probs.copy()

    # -------------------------
    # Final predictions
    # -------------------------

    preds = final_probs.argmax(axis=1)

    # -------------------------
    # Metrics
    # -------------------------

    score = balanced_accuracy_score(
        y_true,
        preds
    )

    cm = confusion_matrix(
        y_true,
        preds
    )

    cm_pct = (
        cm.astype(float)
        / cm.sum(axis=1, keepdims=True)
        * 100
    )

    # -------------------------
    # Print results
    # -------------------------

    print(
        f"{title} Balanced Accuracy: "
        f"{score:.5f}"
    )

    print()

    print(
        classification_report(
            y_true,
            preds,
            target_names=class_names,
            digits=4
        )
    )

    # -------------------------
    # Confusion matrices
    # -------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        annot_kws={"size": 12},
        ax=axes[0]
    )

    axes[0].set_title(
        f"{title} — Raw Counts",
        fontweight="bold"
    )

    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        annot_kws={"size": 12},
        ax=axes[1]
    )

    axes[1].set_title(
        f"{title} — Row % (Recall)",
        fontweight="bold"
    )

    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

    return {
        "balanced_accuracy": score,
        "predictions": preds,
        "confusion_matrix": cm,
        "confusion_matrix_pct": cm_pct,
        "final_probs": final_probs
    }