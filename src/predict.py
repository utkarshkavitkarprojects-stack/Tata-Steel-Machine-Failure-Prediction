# src/predict.py

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import feature_engineering

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "final_feature_columns.pkl")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")


# =========================================================
# LOAD ARTIFACTS
# =========================================================
def load_artifacts():
    """
    Load trained model, feature columns, and threshold config.
    """
    model = joblib.load(MODEL_PATH)
    final_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        threshold = config.get("threshold", 0.40)
    else:
        threshold = 0.40

    return model, final_feature_columns, threshold


# =========================================================
# PREPROCESS INPUT FOR INFERENCE
# =========================================================
def preprocess_input(df):
    """
    Apply inference-safe preprocessing only.
    No train-test split, no SMOTE.
    """
    df = df.copy()

    # Encode categorical columns if present
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


# =========================================================
# ALIGN FEATURES
# =========================================================
def align_features(df, final_feature_columns):
    """
    Ensure processed dataframe matches training feature columns.
    Adds missing columns with 0 and reorders columns.
    """
    df = df.copy()

    for col in final_feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns in exact same order
    df = df[final_feature_columns]

    return df


# =========================================================
# BUSINESS OUTPUT HELPERS
# =========================================================
def get_risk_level(prob):
    if prob < 0.20:
        return "Low Risk"
    elif prob < 0.40:
        return "Moderate Risk"
    elif prob < 0.70:
        return "High Risk"
    else:
        return "Critical Risk"


def get_maintenance_action(prob):
    if prob < 0.20:
        return "Machine appears stable. Continue routine monitoring."
    elif prob < 0.40:
        return "Inspect machine during next scheduled maintenance."
    elif prob < 0.70:
        return "High failure risk. Maintenance team should inspect soon."
    else:
        return "Immediate intervention recommended to avoid breakdown."


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================
def predict_machine_failure(input_df):
    """
    Predict machine failure probability and final classification.

    Parameters
    ----------
    input_df : pd.DataFrame
        Raw machine input dataframe

    Returns
    -------
    pd.DataFrame
        Prediction output with probability, class, and business interpretation
    """
    # Load artifacts
    model, final_feature_columns, threshold = load_artifacts()

    # Keep raw copy
    raw_df = input_df.copy()

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df = feature_engineering(raw_df.copy())

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df = preprocess_input(df)

    # -----------------------------
    # Align Features
    # -----------------------------
    X_final = align_features(df, final_feature_columns)

    # -----------------------------
    # Predict
    # -----------------------------
    failure_proba = model.predict_proba(X_final)[:, 1]
    failure_pred = (failure_proba >= threshold).astype(int)

    # -----------------------------
    # Final Output
    # -----------------------------
    result_df = raw_df.copy()
    result_df["failure_probability"] = np.round(failure_proba, 4)
    result_df["failure_prediction"] = failure_pred
    result_df["risk_level"] = result_df["failure_probability"].apply(get_risk_level)
    result_df["maintenance_action"] = result_df["failure_probability"].apply(get_maintenance_action)

    return result_df


# =========================================================
# SINGLE MACHINE WRAPPER
# =========================================================
def predict_single_machine(machine_dict):
    """
    Predict for one machine record.
    """
    input_df = pd.DataFrame([machine_dict])
    return predict_machine_failure(input_df)

# =========================================================
# CSV BATCH PREDICTION
# =========================================================
def predict_from_csv(csv_path, output_path=None):
    """
    Predict machine failure for batch CSV input.

    Parameters
    ----------
    csv_path : str
        Path to input CSV file
    output_path : str, optional
        Path to save prediction output CSV

    Returns
    -------
    pd.DataFrame
        Prediction results dataframe
    """
    df = pd.read_csv(csv_path)
    result_df = predict_machine_failure(df)

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")

    return result_df


# =========================================================
# TEST RUN
# =========================================================
if __name__ == "__main__":

    sample_machine = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 120
    }

    result = predict_single_machine(sample_machine)

    print("\n🔮 Final Prediction Result:")
    print(result[[
        "failure_probability",
        "failure_prediction",
        "risk_level",
        "maintenance_action"
    ]])

    # -----------------------------
    # Batch CSV Test (Optional)
    # -----------------------------
    # batch_result = predict_from_csv("sample_machines.csv", "predictions.csv")
    # print("\n📂 Batch Prediction Preview:")
    # print(batch_result.head())