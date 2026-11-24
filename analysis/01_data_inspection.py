import pandas as pd

# Load dataset
df = pd.read_csv("dataset/European_Bank_Dataset.csv")

# 1. Display first five records
print("\nFIRST FIVE RECORDS")
print(df.head())

# 2. Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# 3. Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# 4. Dataset information
print("\nDATASET INFORMATION")
df.info()

# 5. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 6. Duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# 7. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 8. Churn distribution
print("\nCUSTOMER CHURN DISTRIBUTION")
print(df["Exited"].value_counts())

print("\nDATA INSPECTION COMPLETED!")