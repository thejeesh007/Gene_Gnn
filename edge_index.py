import pandas as pd
import numpy as np
import torch

# load adjacency
adj = pd.read_csv("adjacency_matrix.csv", index_col=0)

# convert to edge index
edge_index = np.array(np.nonzero(adj.values))

# convert to torch tensor
edge_index = torch.tensor(edge_index, dtype=torch.long)

# save
torch.save(edge_index, "edge_index.pt")

print("Edge index created:", edge_index.shape)