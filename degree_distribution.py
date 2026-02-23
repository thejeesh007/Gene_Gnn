import pandas as pd
import matplotlib.pyplot as plt

adj = pd.read_csv("11.adjacency_matrix.csv", index_col=0)
degree = adj.sum(axis=1)

plt.figure(figsize=(6,4))
plt.hist(degree, bins=40)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution of Gene Network")
plt.tight_layout()
plt.savefig("degree_distribution.png", dpi=300)
plt.show()