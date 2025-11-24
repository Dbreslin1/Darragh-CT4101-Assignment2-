import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rfr_results = pd.read_csv("rfr_grid_search_results.csv")

plt.figure(figsize=(8,6))
sns.lineplot(
    data=rfr_results,
    x='param_n_estimators',
    y='mean_test_score',
    hue='param_max_features',
    marker="o"
)

plt.title("RFR Hyperparameters vs Accuracy")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Mean Test Score (higher = better)")
plt.legend(title="Max Features")
plt.tight_layout()
plt.show()



