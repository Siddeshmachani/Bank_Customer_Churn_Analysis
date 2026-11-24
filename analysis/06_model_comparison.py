import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model comparison results
df = pd.read_csv("dataset/model_comparison_results.csv")

# Create visualization folder
os.makedirs("visualizations", exist_ok=True)

print("MODEL PERFORMANCE COMPARISON")
print(df.round(4).to_string(index=False))

# 1. Accuracy comparison
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="Model", y="Accuracy")
plt.title("Accuracy Comparison of Machine Learning Models")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.savefig(
    "visualizations/08_model_accuracy.png",
    bbox_inches="tight"
)
plt.show()

# 2. Precision, Recall and F1 comparison
metrics = df.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=metrics,
    x="Model",
    y="Score",
    hue="Metric"
)
plt.title("Model Performance Metrics Comparison")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.savefig(
    "visualizations/09_model_metrics.png",
    bbox_inches="tight"
)
plt.show()

print("\nMODEL COMPARISON VISUALIZATIONS COMPLETED!")