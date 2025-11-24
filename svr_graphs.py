import pandas as pd
import matplotlib.pyplot as plt

# Load SVR results
svr_results = pd.read_csv("svr_grid_search_results.csv")

# Create a new column combining hyperparameters for labeling
svr_results['combo'] = svr_results['param_C'].astype(str) + " | gamma=" + svr_results['param_gamma'].astype(str)

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(svr_results['combo'], svr_results['mean_test_score'])
plt.xticks(rotation=45, ha='right')
plt.title("SVR Hyperparameter Combinations vs Accuracy")
plt.xlabel("Hyperparameter Combination (C, gamma)")
plt.ylabel("Mean Test Score (higher = better)")
plt.tight_layout()
plt.show()


