import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
corr = pd.read_csv("10.correlation_matrix.csv", index_col=0)

values = corr.values.flatten()

plt.figure(figsize=(6,4))
plt.hist(values, bins=50)
plt.xlabel("Correlation coefficient")
plt.ylabel("Frequency")
plt.title("Distribution of Pearson Correlation Values")
plt.tight_layout()
plt.savefig("correlation_distribution.png", dpi=300)
plt.show()