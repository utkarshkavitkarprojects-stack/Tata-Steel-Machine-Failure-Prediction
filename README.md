# 🏭 Tata Steel Machine Failure Prediction

An end-to-end **Predictive Maintenance Machine Learning Project** built to predict industrial machine failure using operational sensor data, engineered features, class imbalance handling, model comparison, threshold tuning, explainability, and deployment-ready modular pipeline design.

This project simulates a **real-world manufacturing use case** where early detection of machine failure can reduce:

- unplanned downtime
- maintenance cost
- production disruption
- operational risk

---

## 📌 Problem Statement

In industrial manufacturing environments, unexpected machine failures can lead to major operational and financial losses.

The objective of this project is to build a machine learning system that can:

- predict whether a machine is likely to fail
- identify high-risk machine conditions early
- support **predictive maintenance decisions**
- make the model usable through a **Streamlit web app**

---

## 🎯 Business Objective

Instead of reacting **after failure happens**, this system enables a **proactive maintenance strategy** by predicting failure probability in advance.

### Real-world value:
- reduce reactive maintenance
- improve machine uptime
- prioritize high-risk machines
- support maintenance scheduling
- improve operational efficiency

---

# 🧠 Project Workflow

This project was built as a **full machine learning pipeline**, not just a notebook experiment.

### End-to-end workflow:
1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
5. Class Imbalance Handling
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Threshold Tuning
10. SHAP Explainability
11. Modular ML Pipeline Development
12. Streamlit Deployment

---

# 📂 Dataset Information

The dataset contains machine operational and sensor-level variables such as:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

It also includes failure-related indicators, which were carefully handled to avoid **data leakage**.

---

# 🔍 Exploratory Data Analysis (EDA)

A complete EDA was performed to understand:

- class imbalance in machine failure
- distribution of numerical variables
- feature relationships
- failure patterns
- operational behavior across machine types

### EDA Highlights:
- failure class was highly imbalanced
- operational parameters like torque, rotational speed, and temperature difference showed useful signal
- failure patterns were explored visually before modeling

EDA helped guide:
- feature engineering decisions
- imbalance handling strategy
- model selection direction

---

# ⚙️ Feature Engineering

To improve model performance and capture operational machine behavior, multiple domain-inspired engineered features were created:

### Engineered Features:
- **Temp_Difference** = Process Temperature - Air Temperature
- **Power** = Rotational Speed × Torque
- **Torque_Speed_Ratio** = Torque / Rotational Speed
- **Wear_Level** = binned tool wear category

These features were designed to capture:
- machine load behavior
- thermal stress difference
- operational strain
- wear-based maintenance risk

---

# 🚫 Data Leakage Prevention

Several columns were intentionally removed because they would leak target information or add no predictive value:

### Dropped Columns:
- `id`
- `Product ID`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

These failure-type columns directly reveal failure conditions and would artificially inflate model performance if included.

This was an important step to keep the project **realistic and interview-defensible**.

---

# 🧹 Data Preprocessing

The preprocessing pipeline included:

- duplicate handling
- categorical encoding
- train-test split
- feature alignment for deployment
- robust handling for inference-time inputs

### Additional deployment-safe preprocessing:
The project was structured to handle both:

- **training-time raw data**
- **real-time / app input data**

This prevents common deployment issues like:
- missing columns
- CSV schema mismatch
- inference-time feature alignment failures

---

# ⚖️ Class Imbalance Handling

The target variable (`Machine failure`) was highly imbalanced.

To address this, the training pipeline included:

## ✅ SMOTE (Synthetic Minority Oversampling Technique)

This helped improve the model’s ability to identify actual machine failure cases instead of simply optimizing for majority-class accuracy.

This was important because in predictive maintenance:

> **Missing a true failure is far more costly than a few false alarms.**

---

# 🤖 Models Trained

Multiple machine learning models were trained and compared:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

### Why multiple models?
This project was intentionally designed to compare:
- interpretability
- recall behavior
- robustness
- precision-recall tradeoff
- industrial applicability

---

# 🔧 Hyperparameter Tuning

Model tuning was performed using:

## `RandomizedSearchCV`

Hyperparameter tuning was applied to:
- Logistic Regression
- Random Forest
- XGBoost

This helped optimize model performance in a structured and scalable way.

---

# 📊 Model Evaluation

Models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Additional evaluation artifacts generated:
- Confusion Matrix
- ROC Curve
- Classification Report

This project does **not rely on accuracy alone**, because machine failure prediction is an **imbalanced classification problem**.

---

# 🏆 Final Model Selection

The final production model selected was:

# **XGBoost Classifier**

### Why XGBoost?
Because it provided the best overall balance between:

- precision
- recall
- F1 score
- ranking performance
- robustness on imbalanced industrial data

It performed better than simpler models and gave the most reliable predictive behavior for this use case.

---

# 🎯 Threshold Tuning (Business-Oriented Decision)

Instead of blindly using the default classification threshold of **0.50**, threshold tuning was performed to better align the model with predictive maintenance goals.

### Thresholds tested:
- 0.30
- 0.35
- 0.40
- 0.45
- 0.50
- 0.55
- 0.60

### Final selected threshold:
# **0.40**

### Why threshold tuning mattered:
In this problem, a slightly lower threshold helps catch more potential failures early.

This makes sense in a maintenance setting where:

> **false negatives (missed failures) are more expensive than false positives.**

Threshold tuning made the model more **business-aware**, not just statistically optimized.

---

# 📈 Final Model Performance

### Final deployed setup:
- **Model:** XGBoost
- **Decision Threshold:** 0.40

### Performance at threshold = 0.40
| Metric | Score |
|--------|-------|
| Accuracy | **0.9822** |
| Precision | **0.4367** |
| Recall | **0.4917** |
| F1 Score | **0.4626** |

### Why these results matter:
This model was optimized for a **high-cost industrial failure scenario**, where improving failure capture is more important than maximizing raw accuracy.

---

# 🔍 Model Explainability (SHAP)

To understand **why the model predicts failure**, SHAP analysis was performed.

## SHAP was used for:
- feature importance interpretation
- local prediction explanation
- validating model behavior
- improving trust in model decisions

### SHAP helped answer:
- Which features push a machine toward failure?
- Which operational conditions are most risky?
- Is the model learning meaningful industrial patterns?

This adds strong **explainability value** to the project and makes it more interview-ready than a black-box-only solution.

---

# 🏗️ Modular ML Pipeline (Production-Style Project Structure)

One of the strongest parts of this project is that it was converted into a **modular ML pipeline in VS Code**, making it much closer to a real-world deployable project.

---

## 📁 Project Structure

```bash
Tata-Steel-Machine-Failure-Prediction/
│
├── artifacts/
│   ├── best_model.pkl
│   ├── final_feature_columns.pkl
│   └── config.json
│
├── model_graphs/
│   ├── *.png
│
├── notebooks/
│   └── Tata_Steel_Machine_Failure_Prediction.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── threshold_tuning.py
│   └── predict.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md