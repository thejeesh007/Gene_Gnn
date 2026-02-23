import pandas as pd
import numpy as np

# load correlation matrix
corr = pd.read_csv("correlation_matrix.csv", index_col=0)

# threshold
threshold = 0.8

# create adjacency
adj = (corr.abs() > threshold).astype(int)

# remove self-loops
np.fill_diagonal(adj.values, 0)

# save
adj.to_csv("adjacency_matrix.csv")

print("Adjacency matrix created")