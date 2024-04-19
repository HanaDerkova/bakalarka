import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from definitions import sh_entropy

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("outputs/statistics.tsv", sep='\t')

print(df)

# Assuming "cross_entropy" is the column name
# Replace it with the actual column name if different
# Assuming you want to group by "architecture" and "input_file"
grouped = df.groupby(["architecture", "input_file"])

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for name, group in grouped:
    architecture, input_file = name
    data = np.load(f"data/{input_file}")
    if architecture == "combined":
        group = group.dropna(subset=['cross_entropy'])  # Drop NaN values in cross_entropy column if any
        ax.plot(group["number_of_states"], group["cross_entropy"], marker='o', linestyle='-', label=f"{architecture} - {input_file}")
        s_h = sh_entropy(data)
        ax.axhline(y=s_h, color='red', linestyle='--', label=f"shannon entropy{s_h:.2f}")

ax.set_xlabel("Number of States")
ax.set_ylabel("Cross Entropy")
ax.set_title("Cross Entropy vs Number of States")
ax.legend()
plt.grid(True)
plt.show()