import pandas as pd

df = pd.read_csv("clean_expression.csv")

# forward fill gene symbols
df["Gene symbol"] = df["Gene symbol"].ffill()

# save
df.to_csv("gene_symbol_filled.csv", index=False)

print("Blank probe rows fixed")