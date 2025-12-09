#soc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = np.array([
    12, 15, 14, 11, 13, 16, 12, 14, 15,
    13, 17, 12, 16, 14, 11, 15, 13, 14,
    12, 18, 16, 13, 12, 15, 14, 17, 13,
    16, 12, 14, 15, 13, 16, 12
])
#print(len(data))

# Frequency distribution
freq = pd.Series(data).value_counts().sort_index()
# Empirical probability distribution
prob_dist = freq / len(data)
prob_dist

#print(prob_dist)


plt.figure(figsize=(8,5))
plt.bar(prob_dist.index, prob_dist.values)
plt.xlabel("Daily Customer Arrivals")
plt.ylabel("Probability")
plt.title("Empirical Probability Distribution of Daily Arrivals")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


mean_empirical = np.mean(data)
variance_empirical = np.var(data, ddof=0)  # population variance
mean_empirical, variance_empirical

print("Expected value (mean):", round(mean_empirical,2))
print("Variance:", round(variance_empirical,2))


if prob_dist.idxmax() in [12, 13, 14, 15]:
    shape_comment = "The distribution is unimodal and centered around 13–15, showing slight symmetry."
else:
    shape_comment = "The distribution does not appear symmetric and may have multiple peaks."

print("\nComment on distribution shape:")
print(shape_comment)

#eoc
