import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "gradient_boosting.pkl"
DATA_PATH = BASE_DIR / "dataset" / "feature_engineered_bank_churn.csv"

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# Prepare feature names in the same order used during training
X = df.drop("Exited", axis=1)
feature_columns = X.columns.tolist()

# Title
st.title("🏦 Bank Customer Churn Prediction System")

st.write(
    "An interactive Machine Learning application "
    "to predict bank customer churn risk."
)

st.divider()

# Sidebar
st.sidebar.header("Customer Information")

credit_score = st.sidebar.slider(
    "Credit Score", 300, 900, 650
)

age = st.sidebar.slider(
    "Age", 18, 90, 35
)

tenure = st.sidebar.slider(
    "Tenure (Years)", 0, 10, 5
)

balance = st.sidebar.number_input(
    "Account Balance", min_value=0.0, value=50000.0
)

products = st.sidebar.selectbox(
    "Number of Bank Products", [1, 2, 3, 4]
)

credit_card = st.sidebar.selectbox(
    "Has Credit Card", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

active_member = st.sidebar.selectbox(
    "Active Member", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

salary = st.sidebar.number_input(
    "Estimated Salary", min_value=0.0, value=50000.0
)

geography = st.sidebar.selectbox(
    "Geography", ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender", ["Female", "Male"]
)

# Create prediction input
customer = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": credit_card,
    "IsActiveMember": active_member,
    "EstimatedSalary": salary,
    "Geography_Germany": int(geography == "Germany"),
    "Geography_Spain": int(geography == "Spain"),
    "Gender_Male": int(gender == "Male"),
    "BalanceSalaryRatio": balance / (salary + 1),
    "ProductDensity": products / (tenure + 1),
    "EngagementProduct": active_member * products,
    "AgeTenureInteraction": age * tenure
}

# Arrange features in training order
input_df = pd.DataFrame([customer])
input_df = input_df[feature_columns]

# Main application
st.header("Customer Churn Risk Calculator")

if st.button("Predict Customer Churn Risk", type="primary"):

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Retention Probability",
            f"{(1 - probability) * 100:.2f}%"
        )

    st.progress(float(probability))

    if probability >= 0.5:
        st.error("High Churn Risk: Customer may leave the bank.")
    else:
        st.success("Lower Churn Risk: Customer is less likely to leave.")

st.divider()

# Churn probability distribution
st.header("Churn Probability Distribution")

dataset_probabilities = model.predict_proba(X)[:, 1]

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    dataset_probabilities,
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Number of Customers")
ax.set_title("Distribution of Customer Churn Probabilities")

st.pyplot(fig)
plt.close(fig)

st.divider()

# Feature importance
st.header("Feature Importance Dashboard")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.caption(
    "Feature importance indicates how much the trained "
    "Gradient Boosting model relies on each feature."
)

st.divider()

# What-if scenario simulator
st.header("What-if Scenario Simulator")

st.write(
    "Explore how changing customer engagement or product "
    "usage affects the model's churn prediction."
)

simulated_active = st.selectbox(
    "Simulated Active Membership",
    [0, 1],
    index=active_member,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

simulated_products = st.selectbox(
    "Simulated Number of Products",
    [1, 2, 3, 4],
    index=products - 1
)

if st.button("Run What-if Simulation"):

    simulated_customer = customer.copy()

    simulated_customer["IsActiveMember"] = simulated_active
    simulated_customer["NumOfProducts"] = simulated_products

    simulated_customer["ProductDensity"] = (
        simulated_products / (tenure + 1)
    )

    simulated_customer["EngagementProduct"] = (
        simulated_active * simulated_products
    )

    simulated_df = pd.DataFrame([simulated_customer])
    simulated_df = simulated_df[feature_columns]

    simulated_probability = model.predict_proba(
        simulated_df
    )[0][1]

    st.metric(
        "Simulated Churn Probability",
        f"{simulated_probability * 100:.2f}%"
    )

    st.info(
        "This is a model-based scenario, not a guarantee "
        "that changing customer behaviour will change actual churn."
    )

st.divider()

st.caption(
    "Bank Customer Churn Prediction | Machine Learning Project | Unified Mentor"
)