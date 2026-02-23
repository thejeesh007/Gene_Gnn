import pandas as pd
import torch

# Load normalized dataset
df = pd.read_csv("normalized_dataset.csv")

# Drop non-feature columns
features = df.drop(columns=["Gene symbol", "variance"])

# Convert to tensor
x = torch.tensor(features.values, dtype=torch.float)

print("Node feature shape:", x.shape)

# Save if needed
torch.save(x, "node_features.pt")