import pandas as pd

# Load original dataset
df = pd.read_csv("dataset/European_Bank_Dataset.csv")

print("Original Dataset Shape:", df.shape)

# 1. Remove unnecessary columns
df.drop(columns=["Year", "CustomerId", "Surname"], inplace=True)

# 2. Encode categorical columns
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int
)

# 3. Separate features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# 4. Display cleaned dataset information
print("\nCLEANED DATASET SHAPE:", df.shape)

print("\nCLEANED COLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nTARGET DISTRIBUTION:")
print(y.value_counts())

# 5. Save cleaned dataset
df.to_csv("dataset/cleaned_bank_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")

print("\nDATA PREPROCESSING COMPLETED!")