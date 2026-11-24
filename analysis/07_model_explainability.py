import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

from sklearn.inspection import PartialDependenceDisplay

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# 1. Load feature-engineered dataset
df = pd.read_csv("dataset/feature_engineered_bank_churn.csv")

X = df.drop("Exited", axis=1)

# 2. Load trained Gradient Boosting model
model = joblib.load("models/gradient_boosting.pkl")

print("GRADIENT BOOSTING MODEL LOADED SUCCESSFULLY")

# 3. Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE RANKING")
print(importance_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Gradient Boosting Feature Importance")
plt.tight_layout()

plt.savefig(
    "visualizations/10_feature_importance.png",
    bbox_inches="tight"
)

plt.show()

# 4. SHAP Analysis
# Use a sample to reduce computation time
X_sample = X.sample(
    n=min(1000, len(X)),
    random_state=42
)

print("\nCALCULATING SHAP VALUES...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# SHAP Summary Plot
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.title("SHAP Summary Plot")
plt.tight_layout()

plt.savefig(
    "visualizations/11_shap_summary.png",
    bbox_inches="tight"
)

plt.show()

# SHAP Feature Importance Bar Plot
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    "visualizations/12_shap_importance.png",
    bbox_inches="tight"
)

plt.show()

# 5. Partial Dependence Plots (Manual Calculation)

features_to_plot = [
    "Age",
    "Balance",
    "NumOfProducts",
    "Tenure"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for feature, ax in zip(features_to_plot, axes.flatten()):

    # Create values across the feature's observed range
    values = np.linspace(
        X[feature].quantile(0.05),
        X[feature].quantile(0.95),
        30
    )

    average_probabilities = []

    for value in values:

        # Copy the original dataset
        X_modified = X.copy()

        # Set the selected feature to one value
        X_modified[feature] = value

        # Calculate average churn probability
        probabilities = model.predict_proba(X_modified)[:, 1]

        average_probabilities.append(probabilities.mean())

    # Plot the results
    ax.plot(values, average_probabilities)

    ax.set_title(f"Partial Dependence: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Average Predicted Churn Probability")
    ax.grid(True)

plt.suptitle("Partial Dependence Analysis", fontsize=15)
plt.tight_layout()

plt.savefig(
    "visualizations/13_partial_dependence.png",
    bbox_inches="tight"
)

plt.show()

print("\nPARTIAL DEPENDENCE PLOTS GENERATED SUCCESSFULLY!")

print("\nMODEL EXPLAINABILITY COMPLETED SUCCESSFULLY!")