import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_bank_churn.csv")

print("ORIGINAL DATASET SHAPE:", df.shape)

# 1. Balance-to-Salary Ratio
df["BalanceSalaryRatio"] = (
    df["Balance"] / (df["EstimatedSalary"] + 1)
)

# 2. Product Density Indicator
df["ProductDensity"] = (
    df["NumOfProducts"] / (df["Tenure"] + 1)
)

# 3. Engagement-Product Interaction
df["EngagementProduct"] = (
    df["IsActiveMember"] * df["NumOfProducts"]
)

# 4. Age-Tenure Interaction
df["AgeTenureInteraction"] = (
    df["Age"] * df["Tenure"]
)

# Display new features
print("\nNEW FEATURES CREATED:")
print(df[[
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProduct",
    "AgeTenureInteraction"
]].head())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Save feature-engineered dataset
df.to_csv(
    "dataset/feature_engineered_bank_churn.csv",
    index=False
)

print("\nFEATURE ENGINEERING COMPLETED!")
print("FINAL DATASET SHAPE:", df.shape)