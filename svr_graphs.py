import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

svr_results = pd.read_csv("svr_grid_search_results.csv")

plt.figure(figsize=(8,6))
sns.lineplot(
    data=svr_results,
    x='param_C',
    y='mean_test_score',
    hue='param_gamma',
    marker="o"
)

plt.title("SVR Hyperparameters vs Accuracy")
plt.xlabel("C")
plt.ylabel("Mean Test Score (higher = better)")
plt.legend(title="Gamma")
plt.tight_layout()
plt.show()



