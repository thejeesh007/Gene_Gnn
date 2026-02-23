import pandas as pd

soft_file = "data.soft"     # your input file
csv_file = "output.csv"     # desired output

data_lines = []
capture = False

with open(soft_file, "r", encoding="utf-8") as f:
    for line in f:
        # Expression table usually starts after this marker
        if line.startswith("!dataset_table_begin"):
            capture = True
            continue
        if line.startswith("!dataset_table_end"):
            capture = False
            break
        
        if capture:
            data_lines.append(line.strip().split("\t"))

# Convert to DataFrame
df = pd.DataFrame(data_lines[1:], columns=data_lines[0])

# Save as CSV
df.to_csv(csv_file, index=False)

print("Converted successfully")