import pandas as pd
import numpy as np


def feature_engineering(df):
    """
    Apply feature engineering for Tata Steel Machine Failure Prediction.
    Works safely for both training and deployment/inference.
    """
    df = df.copy()

    # =========================
    # Feature Engineering
    # =========================
    if all(col in df.columns for col in ['Process temperature [K]', 'Air temperature [K]']):
        df['Temp_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

    if all(col in df.columns for col in ['Rotational speed [rpm]', 'Torque [Nm]']):
        df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
        df["Torque_Speed_Ratio"] = df["Torque [Nm]"] / df["Rotational speed [rpm]"].replace(0, np.nan)

    if 'Tool wear [min]' in df.columns:
        df["Wear_Level"] = pd.cut(
            df["Tool wear [min]"],
            bins=[0, 100, 200, 300],
            labels=[0, 1, 2],
            include_lowest=True
        )
        df["Wear_Level"] = df["Wear_Level"].astype(float).fillna(0).astype(int)

    # =========================
    # Drop Leakage / Useless Columns
    # =========================
    df.drop(
        ["id", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        axis=1,
        inplace=True,
        errors='ignore'
    )

    # =========================
    # Clean Column Names
    # =========================
    df.columns = (
        df.columns
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('<', '', regex=False)
        .str.replace('>', '', regex=False)
        .str.replace(' ', '_', regex=False)
    )

    return df