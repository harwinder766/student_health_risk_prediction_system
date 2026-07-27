# Data

This directory contains the data required by the deployed Student Health Risk Prediction application.

## `analytics_data.parquet`

This is a processed dataset used by the **Analytics module** of the Streamlit application.

It contains the features required to generate the dashboard's health, lifestyle, and numerical analyses.

## Model Training Data

The original training and test datasets used for model development were obtained from a **Kaggle competition**.

The raw datasets are not included in this repository due to their size.

These datasets were used for:

- Exploratory Data Analysis (EDA)
- Feature engineering
- CatBoost model development
- XGBoost model development
- LightGBM model development
- Cross-validation and model evaluation
- Blending experiments
- Final model training

## Reproducing the Model Training

To reproduce the model-development notebooks:

1. Download the original competition dataset from Kaggle.The link of the competition is -
https://www.kaggle.com/competitions/playground-series-s6e7

2. Place the required files inside this `data/` directory.

The expected structure is:

    data/
    ├── train.csv
    ├── test.csv
    └── analytics_data.parquet

3. Run the notebooks from the `notebooks/` directory as described in the main project README.

> **Note:** The Kaggle datasets are not redistributed in this repository. Please obtain them directly from the original Kaggle competition.