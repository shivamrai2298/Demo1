#soc
#Part B Q.1 i
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


x0 = 7
m = 10**5       # 100000
a = 5
n = 10_000      


x = np.zeros(n)
x[0] = x0

# MCG recursion
for i in range(1, n):
    x[i] = (a * x[i-1]) % m

u = x / m

# Display first 10 random numbers
print("First 10 generated Uniform(0,1) numbers:")
print(u[:10])


# Histogram to check if it looks Uniform(0,1)
plt.figure(figsize=(8,5))
plt.hist(u, bins=30, density=True, edgecolor="black", alpha=0.7)
plt.title("Histogram of MCG-Generated Random Numbers")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()






#PartB Q1. ii
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("CarData_for_set_1.csv")

# Show top 5 rows
print("Dataset preview:")
print(df.head())

# Select required features
X = df[["mpg", "hp"]]


# Perform K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

print("\nCluster Centers (mpg, hp):")
print(kmeans.cluster_centers_)


# Plot clusters
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, x="mpg", y="hp",
    hue="cluster", palette="Set1", s=120
)
plt.title("K-Means Clustering on Car Data (mpg vs hp)")
plt.xlabel("Miles per Gallon (mpg)")
plt.ylabel("Horsepower (hp)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Interpretation 
print("\nINTERPRETATION:")
print("""
Cluster 0 → Likely high-mpg, low-hp cars 
(Fuel efficient, less powerful)

Cluster 1 → Likely low-mpg, high-hp cars
(High power, fuel-inefficient)
""")

#eoc


