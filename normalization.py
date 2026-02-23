import pandas as pd
from scipy.stats import zscore

# Load your final filtered dataset
df = pd.read_csv("final_filtered_dataset.csv")

# Separate important columns
gene_col = df["Gene symbol"]
variance_col = df["variance"]

# Extract only GSM columns (all except gene + variance)
gsm_cols = df.columns.difference(["Gene symbol", "variance"])
expr = df[gsm_cols]

# Ensure numeric
expr = expr.apply(pd.to_numeric, errors="coerce")

# Apply row-wise z-score normalization
expr_z = expr.apply(zscore, axis=1)

# Combine everything back
df_normalized = pd.concat([gene_col, expr_z, variance_col], axis=1)

# Save normalized dataset
df_normalized.to_csv("normalized_dataset.csv", index=False)

print("Normalization complete.")