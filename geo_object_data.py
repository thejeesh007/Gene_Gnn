from torch_geometric.data import Data
import torch

edge_index = torch.load("edge_index.pt")
x = torch.load("node_features.pt")

data = Data(x=x, edge_index=edge_index)

print(data)