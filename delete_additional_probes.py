import pandas as pd

# load CSV file (with variance column already computed)
df = pd.read_csv("top300_with_variance.csv")

# sort by variance descending
df = df.sort_values("variance", ascending=False)

# remove duplicate probes per gene (keep highest variance probe)
df_unique = df.drop_duplicates(subset=["Gene symbol"], keep="first")

# save final dataset (variance preserved)
df_unique.to_csv("final_filtered_dataset.csv", index=False)

print("Final dataset saved with variance column")