import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_models(X_train, y_train, model_dir="models"):
    """
    Trains and tunes multiple ML models using RandomizedSearchCV.
    
    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        model_dir (str): Directory to save trained models
        
    Returns:
        tuned_models (dict): Dictionary of best tuned models
    """

    os.makedirs(model_dir, exist_ok=True)

    # =========================
    # Model Configurations
    # =========================
    models = {
        "logistic_regression": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ))
            ]),
            "params": {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__solver': ['liblinear', 'lbfgs'],
                'model__penalty': ['l2']
            },
            "n_iter": 10
        },

        "decision_tree": {
            "pipeline": Pipeline([
                ('model', DecisionTreeClassifier(
                    random_state=42
                ))
            ]),
            "params": {
                'model__max_depth': [None, 5, 10, 15, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__criterion': ['gini', 'entropy']
            },
            "n_iter": 10
        },

        "random_forest": {
            "pipeline": Pipeline([
                ('model', RandomForestClassifier(
                    random_state=42
                ))
            ]),
            "params": {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [None, 5, 10, 15, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2']
            },
            "n_iter": 15
        },

        "xgboost": {
            "pipeline": Pipeline([
                ('model', XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    random_state=42
                ))
            ]),
            "params": {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [3, 4, 5, 6, 8],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'model__min_child_weight': [1, 3, 5, 7],
                'model__gamma': [0, 0.1, 0.3, 0.5],
                'model__reg_alpha': [0, 0.01, 0.1, 1],
                'model__reg_lambda': [1, 2, 5, 10]
            },
            "n_iter": 25
        }
    }

    # =========================
    # Train & Tune Models
    # =========================
    tuned_models = {}
    tuning_results = {}

    for model_name, config in models.items():
        print(f"\n🔍 Training and tuning {model_name}...")

        random_search = RandomizedSearchCV(
            estimator=config["pipeline"],
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            scoring='f1',
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        tuned_models[model_name] = random_search.best_estimator_
        tuning_results[model_name] = {
            "best_params": random_search.best_params_,
            "best_cv_f1": random_search.best_score_
        }

        # Save tuned best model
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(random_search.best_estimator_, f)

        # Save tuning results
        results_path = os.path.join(model_dir, f"{model_name}_tuning_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(tuning_results[model_name], f)

        print(f"✅ Best {model_name} saved to: {model_path}")
        print(f"📌 Best Params: {random_search.best_params_}")
        print(f"📌 Best CV F1 Score: {random_search.best_score_:.4f}")

    return tuned_models, tuning_results