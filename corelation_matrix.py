import pandas as pd

# Load normalized dataset
df = pd.read_csv("normalized_dataset.csv")

# Separate gene names
genes = df["Gene symbol"]

# Remove non-expression columns
expr = df.drop(columns=["Gene symbol", "variance"])

# Transpose so genes become columns
corr_matrix = expr.T.corr()

# Save correlation matrix
corr_matrix.to_csv("correlation_matrix.csv", index=True)

print("Correlation matrix created.")

print(corr_matrix.shape)   # should be (340, 340)
print(corr_matrix.iloc[0, 0])  # should be 1 (self-correlation)