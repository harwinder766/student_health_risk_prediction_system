import numpy as np
import optuna
from sklearn.metrics import balanced_accuracy_score


def optimize_class_weights(probs,y_true,n_trials=500,weight_range=(0.25, 4.0),seed=42):

    n_classes = probs.shape[1]

    raw_pred = probs.argmax(axis=1)

    raw_score = balanced_accuracy_score(
        y_true,
        raw_pred
    )


    def objective(trial):

        # Fix first class at 1.0
        weights = [1.0]

        for i in range(1, n_classes):

            w = trial.suggest_float(
                f"w_{i}",
                weight_range[0],
                weight_range[1],
                log=True
            )

            weights.append(w)

        weights = np.array(weights)

        pred = (
            probs * weights
        ).argmax(axis=1)

        score = balanced_accuracy_score(
            y_true,
            pred
        )

        return score

    sampler = optuna.samplers.TPESampler(
        seed=seed
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    best_weights = np.ones(n_classes)

    for i in range(1, n_classes):

        best_weights[i] = (
            study.best_params[f"w_{i}"]
        )

    best_score = study.best_value

    print(
        f"Raw score       : {raw_score:.5f}"
    )

    print(
        f"Optimized score : {best_score:.5f}"
    )

    print(
        f"Improvement     : "
        f"{best_score - raw_score:+.5f}"
    )

    print(
        f"Weights         : "
        f"{np.round(best_weights, 4)}"
    )

    return best_weights, best_score