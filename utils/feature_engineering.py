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
