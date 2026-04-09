import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

# =========================
# Output Directories
# =========================
graph_dir = "outputs/model_graphs"
report_dir = "outputs/reports"

os.makedirs(graph_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots and saves confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    path = os.path.join(graph_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()

    return path


def plot_roc_curve(y_true, y_prob, model_name):
    """
    Plots and saves ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    path = os.path.join(graph_dir, f"{model_name}_roc_curve.png")
    plt.savefig(path)
    plt.close()

    return path


def save_classification_report(y_true, y_pred, model_name):
    """
    Saves classification report as a text file.
    """
    report = classification_report(y_true, y_pred, zero_division=0)

    report_path = os.path.join(report_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def evaluate_models(X_test, y_test, model_path='models'):
    """
    Loads all candidate trained models, evaluates them on test data,
    saves plots/reports, and selects the best model based on F1 score.

    Parameters:
        X_test (pd.DataFrame or np.array): Test features
        y_test (pd.Series or np.array): Test target
        model_path (str): Path where trained models are saved

    Returns:
        results (dict): Evaluation metrics for all models
        best_model: Best-performing model object
        best_model_name (str): Name of best model
    """

    # Exclude already saved best_model.pkl from candidate evaluation
    model_files = [
        f for f in os.listdir(model_path)
        if f.endswith('_model.pkl') and f != 'best_model.pkl'
    ]

    results = {}
    models_dict = {}

    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
        print(f"\n📦 Loading {model_name}...")

        model = joblib.load(os.path.join(model_path, model_file))
        models_dict[model_name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_prob)

        print(f"📊 Accuracy: {acc:.4f}")
        print(f"📊 Precision: {prec:.4f}")
        print(f"📊 Recall: {rec:.4f}")
        print(f"📊 F1 Score: {f1:.4f}")
        print(f"📊 ROC-AUC: {auc_score:.4f}")

        print("📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(y_test, y_prob, model_name)
        save_classification_report(y_test, y_pred, model_name)

        results[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_score
        }

    print("\n✅ All models evaluated!")

    # =========================
    # Select Best Model Based on F1 Score
    # =========================
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models_dict[best_model_name]

    best_model_path = os.path.join(model_path, "best_model.pkl")
    joblib.dump(best_model, best_model_path)

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"✅ Saved best model to: {best_model_path}")

    return results, best_model, best_model_name