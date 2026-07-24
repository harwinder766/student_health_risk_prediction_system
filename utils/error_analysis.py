import pandas as pd


def analyze_errors(
    df,
    y_true,
    y_pred,
    label_encoder,
    analysis_cols=None,
    top_confusions=6
):
    """
    Basic classification error analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Original feature dataframe.

    y_true : array-like
        True encoded labels.

    y_pred : array-like
        Predicted encoded labels.

    label_encoder : fitted LabelEncoder
        Encoder used for the target.

    analysis_cols : list, optional
        Columns for subgroup error analysis.

    top_confusions : int
        Number of confusion pairs to display.

    Returns
    -------
    error_df : pd.DataFrame
        Full dataframe containing predictions and correctness.

    errors : pd.DataFrame
        Only incorrectly classified rows.
    """

    error_df = df.copy()

    error_df["y_true"] = label_encoder.inverse_transform(y_true)
    error_df["y_pred"] = label_encoder.inverse_transform(y_pred)

    error_df["correct"] = (
        error_df["y_true"] == error_df["y_pred"]
    )

    errors = error_df[~error_df["correct"]].copy()

    # -------------------------
    # Overall errors
    # -------------------------

    error_rate = len(errors) / len(error_df) * 100

    print(
        f"Total errors: {len(errors):,} "
        f"({error_rate:.2f}%)"
    )

    # -------------------------
    # Errors by true class
    # -------------------------

    print("\nErrors by TRUE class:")

    print(
        errors
        .groupby("y_true")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    # -------------------------
    # Confusion pairs
    # -------------------------

    print("\nTop confusion pairs (true → predicted):")

    print(
        errors
        .groupby(["y_true", "y_pred"])
        .size()
        .sort_values(ascending=False)
        .head(top_confusions)
        .to_string()
    )

    # -------------------------
    # Subgroup analysis
    # -------------------------

    if analysis_cols:

        for col in analysis_cols:

            print(f"\nErrors by {col}:")

            print(
                errors[col]
                .astype(str)
                .value_counts()
                .to_string()
            )

    return error_df, errors