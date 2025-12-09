#soc
#Part B Question2

import pandas as pd

x = np.array([4.12, 4.25, 3.98, 4.30, 4.18, 4.05, 4.22, 4.10, 4.28, 4.15])

# a) Sample mean
sample_mean = np.mean(x)
print("Sample Mean:", sample_mean)

# ---- Bootstrap (Bias of mean) ----
np.random.seed(456)
B = 1000
n = len(x)

boot_means = np.zeros(B)

for b in range(B):
    boot_sample = np.random.choice(x, size=n, replace=True)
    boot_means[b] = np.mean(boot_sample)

# Bootstrap estimate of bias
bias = np.mean(boot_means) - sample_mean

print("Bootstrap Mean of Means:", np.mean(boot_means))
print("Bootstrap Bias Estimate:", bias)

# b) Bootstrap Standard Error of the sample mean
bootstrap_se = np.std(boot_means, ddof=1)
print("Bootstrap SE of Sample Mean:", bootstrap_se)


#eoc
