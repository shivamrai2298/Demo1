#soc
#Part C Question1

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

np.random.seed(456)

def importance_estimate(N=100000):
    X = np.random.exponential(scale=1.0, size=N)  # Exp(1)
    weights = 1.0 / (1.0 + X**2)                   # w(X)
    estimate = weights.mean()
    se = weights.std(ddof=1) / sqrt(N)
    return estimate, se, X, weights

N = 100000
estimate, se, X, weights = importance_estimate(N=N)

print(f"Importance sampling estimate (N={N}): {estimate:.8f}")
print(f"Estimated standard error: {se:.8f}")

# numerical integration using scipy if available
from scipy import integrate
true_val = integrate.quad(lambda t: np.exp(-t)/(1+t**2), 0, np.inf)[0]
print(f"Numerical integration (quad) value: {true_val:.8f}")
print(f"Absolute error: {abs(estimate-true_val):.8e}")

# nonparametric bootstrap of the weights to visualise variability
B = 2000
boot_means = np.empty(B)
rng = np.random.RandomState(123)
for b in range(B):
    idx = rng.choice(len(weights), size=len(weights), replace=True)
    boot_means[b] = weights[idx].mean()

print(f"Bootstrap mean of estimates: {boot_means.mean():.8f}")
print(f"Bootstrap sd (bootstrap se): {boot_means.std(ddof=1):.8f}")

plt.hist(boot_means, bins=50)
plt.title("Bootstrap distribution of importance-sampling estimate")
plt.xlabel("Estimate")
plt.ylabel("Frequency")
plt.show()



#eoc

