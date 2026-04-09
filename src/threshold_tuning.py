import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def threshold_tuning(X_test, y_test, model_path='models/best_model.pkl'):
    """
    Evaluates multiple classification thresholds on the best model
    and selects a final threshold based on business decision.
    """

    # =========================
    # Load Best Model
    # =========================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Best model not found at: {model_path}")

    best_model = joblib.load(model_path)
    print(f"📦 Loaded best model from: {model_path}")

    # =========================
    # Predict Probabilities
    # =========================
    y_proba_final = best_model.predict_proba(X_test)[:, 1]

    # =========================
    # Threshold Search
    # =========================
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    threshold_results = []

    for threshold in thresholds:
        y_pred_thresh = (y_proba_final >= threshold).astype(int)

        threshold_results.append({
            'Threshold': threshold,
            'Accuracy': round(accuracy_score(y_test, y_pred_thresh), 4),
            'Precision': round(precision_score(y_test, y_pred_thresh, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred_thresh, zero_division=0), 4),
            'F1_Score': round(f1_score(y_test, y_pred_thresh, zero_division=0), 4)
        })

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df = threshold_df.sort_values(by='F1_Score', ascending=False).reset_index(drop=True)

    # =========================
    # Display Results
    # =========================
    print("\n🎯 Threshold Tuning Results:")
    print(threshold_df)

    # =========================
    # Final Threshold Selection
    # =========================
    # Final threshold fixed from notebook business decision
    best_threshold = 0.40

    print(f"\n🏆 Final Selected Threshold (fixed from notebook analysis): {best_threshold}")

    return threshold_df, best_threshold

    