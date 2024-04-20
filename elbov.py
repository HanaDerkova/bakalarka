import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import sh_entropy


# Specify the directory path
folder_path = 'outputs/no_mean'

# List to store file paths
file_paths = []

# Range of IDs from 63 to 92
for id in range(63, 93):
    file_path = os.path.join(folder_path, f'{id}', 'cross_entropy.csv')
    file_paths.append(file_path)


experiments_list="config_folder/experiments.csv"

cross_entropies = pd.concat([pd.read_csv(f, sep=",") for f in file_paths])
experiments = pd.read_csv(experiments_list, sep=",")
cross_entropies = cross_entropies.merge(experiments, on="order")

cross_entropies.to_csv("no_mean_output.csv", sep="\t", index=False)


# Load the CSV file into a pandas DataFrame
df = pd.read_csv("outputs/statistics.tsv", sep='\t')
df_1 = pd.read_csv("no_mean_output.csv", sep='\t')

# Assuming "cross_entropy" is the column name
# Replace it with the actual column name if different
# Assuming you want to group by "architecture" and "input_file"
grouped = df.groupby(["architecture", "input_file", "no_mean"])
grouped_1 = df_1.groupby(["architecture", "input_file", "no_mean"])

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# for name, group in grouped_1:
#     architecture, input_file = name
#     data = np.load(f"data/{input_file}")
#     if architecture == "chain":
#         group = group.dropna(subset=['cross_entropy'])  # Drop NaN values in cross_entropy column if any
#         ax.plot(group["number_of_states"], group["cross_entropy"], marker='o', linestyle='-', label=f"NO MEAN {architecture} - {input_file}")
#         s_h = sh_entropy(data)
#         ax.axhline(y=s_h, color='red', linestyle='--', label=f"shannon entropy {s_h:.2f}")

print(grouped)

for name, group in grouped:
    print(len(name))
    #architecture, input_file = name
    data = np.load(f"data/{input_file}")
    if architecture == "chain" and input_file == "bimodal.npy":
        group = group.dropna(subset=['cross_entropy'])  # Drop NaN values in cross_entropy column if any
        ax.plot(group["number_of_states"], group["cross_entropy"], marker='o', linestyle='-', label=f"MEAN SHIFT. {architecture} - {input_file}")
        s_h = sh_entropy(data)
        #ax.axhline(y=s_h, color='red', linestyle='--', label=f"shannon entropy {s_h:.2f}")


ax.set_xlabel("Number of States")
ax.set_ylabel("Cross Entropy")
ax.set_title("Cross Entropy vs Number of States")
ax.legend()
plt.grid(True)
plt.show()