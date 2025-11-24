import pandas as pd
import matplotlib.pyplot as plt

# Load RFR results
rfr_results = pd.read_csv("rfr_grid_search_results.csv")

# Create a new column combining hyperparameters for labeling
rfr_results['combo'] = rfr_results['param_n_estimators'].astype(str) + " | max_features=" + rfr_results['param_max_features'].astype(str)

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(rfr_results['combo'], rfr_results['mean_test_score'])
plt.xticks(rotation=45, ha='right')
plt.title("RFR Hyperparameter Combinations vs Accuracy")
plt.xlabel("Hyperparameter Combination (n_estimators, max_features)")
plt.ylabel("Mean Test Score (higher = better)")
plt.tight_layout()
plt.show()

