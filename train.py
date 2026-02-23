import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from gat_model import GAT
from torch_geometric.data import Data

# Load data
edge_index = torch.load("12.edge_index.pt")
x = torch.load("13.node_features.pt")

data = Data(x=x, edge_index=edge_index)

# Initialize model
model = GAT(in_channels=11, hidden_channels=32, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# ---------------- TRAINING ----------------
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index)

    pos_edge_index = data.edge_index

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )

    pos_score = decode(z, pos_edge_index)
    neg_score = decode(z, neg_edge_index)

    pos_label = torch.ones(pos_score.size(0))
    neg_label = torch.zeros(neg_score.size(0))

    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_score, neg_score]),
        torch.cat([pos_label, neg_label])
    )

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ---------------- EVALUATION ----------------
model.eval()

with torch.no_grad():
    z = model(data.x, data.edge_index)

    pos_score = decode(z, data.edge_index)

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1)
    )

    neg_score = decode(z, neg_edge_index)

    print("Embedding shape:", z.shape)
    print("Positive edge score mean:", pos_score.mean().item())
    print("Negative edge score mean:", neg_score.mean().item())