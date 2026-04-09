import streamlit as st
import pandas as pd
import numpy as np
import os

from src.predict import predict_single_machine, predict_machine_failure

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Tata Steel Machine Failure Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            background-color: #1c1f26;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .section-title {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #ffffff;
        }
        .subtle-text {
            color: #A9A9A9;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("🏭 Tata Steel Machine Failure Prediction Dashboard")
st.markdown(
    """
    Predict potential machine failures using machine sensor parameters and engineered operational features.  
    This dashboard supports **single machine prediction** and **batch CSV prediction** for predictive maintenance use cases.
    """
)

st.markdown("---")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Single Prediction", "📂 Batch Prediction", "ℹ️ About Project"]
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def show_risk_badge(prob):
    if prob < 0.30:
        st.success("🟢 Low Risk")
    elif prob < 0.60:
        st.warning("🟡 Medium Risk")
    elif prob < 0.80:
        st.warning("🟠 High Risk")
    else:
        st.error("🔴 Very High Risk")


def show_maintenance_action(prob):
    if prob < 0.30:
        st.success("✅ No immediate maintenance action required.")
    elif prob < 0.60:
        st.warning("🛠️ Monitor machine and schedule preventive inspection.")
    elif prob < 0.80:
        st.warning("⚠️ High failure risk detected. Maintenance recommended soon.")
    else:
        st.error("🚨 Critical machine condition. Immediate maintenance required!")


# =========================================================
# PAGE 1: SINGLE PREDICTION
# =========================================================
if page == "🔮 Single Prediction":

    st.subheader("🔮 Single Machine Failure Prediction")
    st.markdown("Enter machine operating parameters below to predict failure probability.")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature [K]", min_value=250.0, max_value=400.0, value=298.1, step=0.1)
        process_temp = st.number_input("Process Temperature [K]", min_value=250.0, max_value=400.0, value=308.6, step=0.1)

    with col_input2:
        rpm = st.number_input("Rotational Speed [rpm]", min_value=1, max_value=10000, value=1551, step=1)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=500.0, value=42.8, step=0.1)
        tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=500, value=120, step=1)

    if st.button("🚀 Predict Machine Failure", use_container_width=True):
        try:
            machine_input = {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }

            result = predict_single_machine(machine_input)

            prob = float(result["failure_probability"].iloc[0])
            pred = int(result["failure_prediction"].iloc[0])
            risk_level = result["risk_level"].iloc[0]
            action = result["maintenance_action"].iloc[0]

            st.markdown("---")
            st.subheader("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Failure Probability", f"{prob:.2%}")

            with col2:
                st.metric("Prediction", "⚠️ Failure Likely" if pred == 1 else "✅ Normal")

            with col3:
                st.metric("Risk Level", risk_level)

            st.markdown("### 🚦 Risk Assessment")
            show_risk_badge(prob)

            st.markdown("### 🛠️ Recommended Maintenance Action")
            st.info(action)

            st.markdown("### 📋 Input Summary")
            st.dataframe(pd.DataFrame([machine_input]), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# =========================================================
# PAGE 2: BATCH PREDICTION
# =========================================================
elif page == "📂 Batch Prediction":

    st.subheader("📂 Batch Machine Failure Prediction")
    st.markdown("Upload a CSV file containing multiple machine records for batch prediction.")

    # -----------------------------
    # Sample CSV Download
    # -----------------------------
    sample_df = pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [298.1, 300.0, 302.5],
        "Process temperature [K]": [308.6, 310.2, 312.1],
        "Rotational speed [rpm]": [1551, 1450, 1600],
        "Torque [Nm]": [42.8, 50.2, 38.5],
        "Tool wear [min]": [120, 180, 75]
    })

    csv_sample = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Sample CSV",
        csv_sample,
        file_name="sample_machine_input.csv",
        mime="text/csv"
    )

    st.markdown("### 📤 Upload CSV")
    uploaded_file = st.file_uploader("Upload machine data CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.markdown("### 👀 Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            result_df = predict_machine_failure(batch_df)

            st.markdown("### ✅ Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            # Summary metrics
            total_records = len(result_df)
            predicted_failures = int(result_df["failure_prediction"].sum())
            avg_failure_prob = float(result_df["failure_probability"].mean())

            st.markdown("### 📊 Batch Summary")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Total Machines", total_records)

            with c2:
                st.metric("Predicted Failures", predicted_failures)

            with c3:
                st.metric("Average Failure Probability", f"{avg_failure_prob:.2%}")

            # Download predictions
            csv_output = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prediction Results",
                csv_output,
                file_name="machine_failure_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Batch prediction failed: {e}")

# =========================================================
# PAGE 3: ABOUT PROJECT
# =========================================================
elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### 🏭 Project Objective
    This project predicts **machine failure risk** using operational and sensor-based machine parameters.

    ### 🎯 Business Goal
    The purpose is to support **predictive maintenance** by identifying machines likely to fail **before actual breakdown happens**, reducing:
    - unplanned downtime
    - maintenance cost
    - production disruption
    - safety risks

    ### ⚙️ Features Used
    - Type
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear
    - Temp Difference *(engineered)*
    - Power *(engineered)*
    - Torque-Speed Ratio *(engineered)*
    - Wear Level *(engineered)*

    ### 🤖 Models Trained
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

    ### 🏆 Final Model
    The final selected model is based on **XGBoost**, with threshold tuning for business-aligned predictive maintenance decisions.

    ### 📌 Use Cases
    - Steel manufacturing plants
    - Industrial machine monitoring
    - Preventive maintenance scheduling
    - Failure risk prioritization
    """)