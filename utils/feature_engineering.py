import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    # ---- Gate interactions (the key features) ----
    df['stress_x_activity'] = (df['stress_level'].fillna('na') + '_' +
                               df['physical_activity_level'].fillna('na'))
    df['stress_x_sleepq']   = (df['stress_level'].fillna('na') + '_' +
                               df['sleep_quality'].fillna('na'))
    df['stress_x_smoke']    = (df['stress_level'].fillna('na') + '_' +
                               df['smoking_alcohol'].fillna('na'))
    df['sleep_x_stress']    = (df['sleep_quality']+ '_' + df['stress_level'])

    # ---- Numeric boundary helpers ----
    df['sleep_lt6'] = (df['sleep_duration'] < 6).astype(float)
    df['sleep_lt6'] = df['sleep_lt6'].where(df['sleep_duration'].notna())
    df['sleep_ge7'] = (df['sleep_duration'] >= 7).astype(float)
    df['sleep_ge7'] = df['sleep_ge7'].where(df['sleep_duration'].notna())

    df['steps_x_exercise']    = df['step_count'] * df['exercise_duration']
    df['activity_efficiency'] = df['calorie_expenditure'] / df['step_count'].replace(0, np.nan)
    df['sleep_x_steps']       = df['sleep_duration'] * df['step_count']
    df['bmi_band']  = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    df['sleep_dev'] = (df['sleep_duration'] - 7.5).abs()

    return df

def fit_target_encoding(
    df,
    cat_cols,
    target
):
    """
    Learn frequency and multiclass target-encoding mappings
    from training data only.
    """

    encoding_info = {}

    classes = sorted(df[target].unique())

    global_probs = (
        df[target]
        .value_counts(normalize=True)
        .to_dict()
    )

    for col in cat_cols:

        # Frequency encoding
        freq_map = (
            df[col]
            .value_counts(normalize=True)
            .to_dict()
        )

        # Target probabilities
        probs = (
            df.groupby(col)[target]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # Make sure every class exists
        for cls in classes:
            if cls not in probs.columns:
                probs[cls] = 0.0

        target_maps = {
            cls: probs[cls].to_dict()
            for cls in classes
        }

        encoding_info[col] = {
            "frequency": freq_map,
            "target_maps": target_maps
        }

    encoding_info["_classes"] = classes
    encoding_info["_global_probs"] = global_probs

    return encoding_info

def transform_target_encoding(
    df,
    encoding_info
):
    """
    Apply previously learned encoding mappings.
    """

    df = df.copy()

    classes = encoding_info["_classes"]
    global_probs = encoding_info["_global_probs"]

    for col, info in encoding_info.items():

        if col.startswith("_"):
            continue

        # Frequency
        df[f"{col}_freq"] = (
            df[col]
            .map(info["frequency"])
            .fillna(0)
        )

        # Class probabilities
        for cls in classes:

            mapping = info["target_maps"][cls]

            df[f"{col}_prob_{cls}"] = (
                df[col]
                .map(mapping)
                .fillna(global_probs.get(cls, 0))
            )

    return df

def prepare_fold(
    train_df,
    test_df,
    train_idx,
    valid_idx,
    cat_cols,
    target
):
    """
    Prepare one CV fold using leakage-safe target encoding.
    """

    train_fold = train_df.iloc[train_idx].copy()
    valid_fold = train_df.iloc[valid_idx].copy()
    test_fold = test_df.copy()

    # Fit ONLY on training fold
    encoding_info = fit_target_encoding(
        train_fold,
        cat_cols,
        target
    )

    # Apply same mappings everywhere
    train_fold = transform_target_encoding(
        train_fold,
        encoding_info
    )

    valid_fold = transform_target_encoding(
        valid_fold,
        encoding_info
    )

    test_fold = transform_target_encoding(
        test_fold,
        encoding_info
    )

    return (
        train_fold,
        valid_fold,
        test_fold,
        encoding_info
    )