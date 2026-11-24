import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualisation style
sns.set_theme(style="whitegrid")

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_bank_churn.csv")

print("EDA STARTED")
print("Dataset Shape:", df.shape)

# --------------------------------
# 1. Customer Churn Distribution
# --------------------------------

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Exited")
plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.savefig("visualizations/01_churn_distribution.png", bbox_inches="tight")
plt.show()

# --------------------------------
# 2. Churn by Age
# --------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Age",
    hue="Exited",
    bins=30,
    multiple="stack"
)
plt.title("Customer Churn by Age")
plt.savefig("visualizations/02_churn_by_age.png", bbox_inches="tight")
plt.show()

# --------------------------------
# 3. Churn by Geography
# --------------------------------

geo_df = pd.read_csv("dataset/European_Bank_Dataset.csv")

plt.figure(figsize=(8, 5))
sns.barplot(data=geo_df, x="Geography", y="Exited")
plt.title("Churn Rate by Geography")
plt.ylabel("Average Churn Rate")
plt.savefig("visualizations/03_churn_by_geography.png", bbox_inches="tight")
plt.show()

# --------------------------------
# 4. Churn by Gender
# --------------------------------

plt.figure(figsize=(7, 5))
sns.barplot(data=geo_df, x="Gender", y="Exited")
plt.title("Churn Rate by Gender")
plt.ylabel("Average Churn Rate")
plt.savefig("visualizations/04_churn_by_gender.png", bbox_inches="tight")
plt.show()

# --------------------------------
# 5. Churn by Account Balance
# --------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=geo_df, x="Exited", y="Balance")
plt.title("Account Balance vs Customer Churn")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.savefig("visualizations/05_balance_vs_churn.png", bbox_inches="tight")
plt.show()

# --------------------------------
# 6. Churn by Number of Products
# --------------------------------

plt.figure(figsize=(8, 5))
sns.barplot(data=geo_df, x="NumOfProducts", y="Exited")
plt.title("Churn Rate by Number of Bank Products")
plt.ylabel("Average Churn Rate")
plt.savefig("visualizations/06_churn_by_products.png", bbox_inches="tight")
plt.show()

# --------------------------------
# 7. Churn by Active Membership
# --------------------------------

plt.figure(figsize=(7, 5))
sns.barplot(data=geo_df, x="IsActiveMember", y="Exited")
plt.title("Churn Rate by Active Membership")
plt.ylabel("Average Churn Rate")
plt.savefig("visualizations/07_churn_by_activity.png", bbox_inches="tight")
plt.show()

# --------------------------------
# Churn Rate Summary
# --------------------------------

print("\nCHURN RATE SUMMARY")

print("\nBy Geography:")
print(geo_df.groupby("Geography")["Exited"].mean().mul(100).round(2))

print("\nBy Gender:")
print(geo_df.groupby("Gender")["Exited"].mean().mul(100).round(2))

print("\nBy Number of Products:")
print(geo_df.groupby("NumOfProducts")["Exited"].mean().mul(100).round(2))

print("\nBy Active Membership:")
print(geo_df.groupby("IsActiveMember")["Exited"].mean().mul(100).round(2))

print("\nEDA COMPLETED SUCCESSFULLY!")