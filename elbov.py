import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import sh_entropy, extract_feature_lengths, mch_to_likelyhood_old, generate_matrix, obj_func_matrix
from PIL import Image


# Define your titles in a dictionary
titles = {
    'gaussian.npy': 'Gaussian small samples',
    'far_gaussian.npy': 'Gaussian big samples',
    'uniform.npy': 'Uniform',
    'bimodal.npy' : 'Bimodal',
    'drosophila_melanogaster_genes_exons.bed' : "Drosophila",
    'Gains_inc_chr.txt' : "Gains inc chr",
    'H3K4me3_chr.txt' : "H3K4me3 chr",
    "hirt_chr.txt" : "hirt chr",
    "promoter_2k_chr_no_chrM.txt": "promoter 2k chr no chrM" 

}

architecture_names = {
    "chain" : 'Self-loop',
    "escape_chain" : 'Early-escape',
    "combined" : 'Combined'
}

outputs_dir = "outputs"

def plot_cross_entropies(architecture, mean_shift, best_cross_e):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("outputs/statistics.tsv", sep='\t')
    grouped = df.groupby(["architecture", "input_file", "no_mean", 'fit'])
    #input_files = df["input_file"].unique()
    input_files = ['gaussian.npy', "far_gaussian.npy", "bimodal.npy", "uniform.npy"]
    #input_files = ['drosophila_melanogaster_genes_exons.bed','Gains_inc_chr.txt', 'H3K4me3_chr.txt', "hirt_chr.txt", "promoter_2k_chr_no_chrM.txt" ]
    # print(best_cross_e)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for ax, input_file in zip(axs.flatten(), input_files):
        for name, group in grouped:
            arch, file, mean_sh, fit = name
            if file == input_file and arch == architecture and mean_sh == mean_shift:
                group = group.dropna(subset=['cross_entropy'])  # Drop NaN values in cross_entropy column if any
                group = group[group["number_of_states"] <= 30] 
                ax.plot(group["number_of_states"], group["cross_entropy"], marker='o', linestyle='-', label="achieved cross entropy")
                # Filter the DataFrame for the specific input file
                filtered_data = best_cross[best_cross["input_file"] == input_file]

                # Plotting
                ax.plot(filtered_data["number_of_states"], filtered_data["best_cross_entropy"], linestyle='-', label="previous best")

                if fit == "d":
                    data = np.load(f"data/{input_file}")
                elif fit == 'i':
                    data = extract_feature_lengths(f'data/{input_file}')
                s_h = sh_entropy(data)
                ax.axhline(y=s_h, color='red', linestyle='--', label=f"Shannon Entropy {s_h:.2f}")
                
                # Update statistics DataFrame
                for idx, row in group.iterrows():
                    input_file = row["input_file"]
                    num_states = row["number_of_states"]
                    cross_entropy = row["cross_entropy"]
                    
                    # Check if there is an existing entry for this input file and number of states
                    existing_entry = best_cross_e[(best_cross_e["input_file"] == input_file) & (best_cross_e["number_of_states"] == num_states)]

                    # If no existing entry or the new cross-entropy is lower, update the DataFrame
                    if existing_entry.empty or cross_entropy < existing_entry["best_cross_entropy"].values[0]:
                        if existing_entry.empty:
                            new_row = {"input_file": input_file, "number_of_states": num_states, "best_cross_entropy": cross_entropy}
                            best_cross_e = pd.concat([best_cross_e, pd.DataFrame([new_row])], ignore_index=True)
                        else:
                            best_cross_e.loc[(best_cross_e["input_file"] == input_file) & (best_cross_e["number_of_states"] == num_states), "best_cross_entropy"] = cross_entropy


        ax.set_xlabel("Number of States")
        ax.set_ylabel("Cross Entropy")
        #ax.legend()
        ax.set_title( f"{architecture_names[architecture]} - {titles[input_file]} " )
        ax.grid(True)
        
    plt.tight_layout()
    if mean_shift == 1:
        plt.savefig(f'pic_visualize/cross_entropy_{architecture}.svg', format='svg')
    else : 
        plt.savefig(f'pic_visualize/cross_entropy_mean_sh_{architecture}.svg', format='svg')
    
    return best_cross_e

def plot_time(architecture, mean_shift):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("outputs/statistics.tsv", sep='\t')
    grouped = df.groupby(["architecture", "input_file", "no_mean"])
    input_files = df["input_file"].unique()
    print(input_files)

    fig, axs = plt.subplots(2, 3, figsize=(12, 10))

    for ax, input_file in zip(axs.flatten(), input_files):
        for name, group in grouped:
            arch, file, mean_sh = name
            if file == input_file and arch == architecture and mean_sh == mean_shift:
                group = group.dropna(subset=['time'])  # Drop NaN values in cross_entropy column if any
                ax.plot(group["number_of_states"], group["time"], marker='o', linestyle='-', label=f"{architecture} - {input_file}")
                

        ax.set_xlabel("Number of States")
        ax.set_ylabel("Time")
        ax.set_title( f"{titles[input_file]}  {architecture}  mean sh :  {mean_shift}" )
        ax.grid(True)
 
    plt.tight_layout()
    plt.savefig(f'pic_visualize/time_{architecture}.svg', format='svg')


def plot_mem_usage(architecture, mean_shift):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("outputs/statistics.tsv", sep='\t')
    grouped = df.groupby(["architecture", "input_file", "no_mean"])
    input_files = df["input_file"].unique()
    print(input_files)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))


    for ax, input_file in zip(axs.flatten(), input_files):
        for name, group in grouped:
            arch, file, mean_sh = name
            if file == input_file and arch == architecture and mean_sh == mean_shift:
                group = group.dropna(subset=['memory'])  # Drop NaN values in cross_entropy column if any
                ax.plot(group["number_of_states"], group["memory"], marker='o', linestyle='-', label=f"{architecture} - {input_file}")

        ax.set_xlabel("Number of States")
        ax.set_ylabel("memory")
        ax.set_title( f"{titles[input_file]}  {architecture}  mean sh :  {mean_shift}" )
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'pic_visualize/mem_{architecture}.svg', format='svg')


def load_trained_model(ax, dir_path, data_file, fit):
    # Read the text file
    with open(f'{dir_path}/trained_model.txt', 'r') as file:
        content = file.readlines()

    # Initialize variables to store parameters
    architecture = None
    number_of_states = None
    parameters = []

    # Iterate through each line of the content
    for idx, line in enumerate(content):
        # Split the line by ':'
        if ':' in line:
            key, value = line.strip().split(':')
            # Remove whitespace from key and value
            key = key.strip()
            value = value.strip()

            # Check the key and assign values accordingly
            if key == 'Architecture':
                architecture = value
            elif key == 'Number of States':
                number_of_states = int(value)
            elif key == 'Parameters':
                # Extract parameters
                for param_line in content[idx+1:]:
                    param = param_line.strip()
                    if param:  # Check if param is not an empty string
                        parameters.append(float(param))
                    else:
                        break  # Stop if an empty line is encountered

    # for now when bigge files vizualize is gonna change
    if fit == "d":
        data = np.load(f"data/{data_file}")
    elif fit == 'i':
        data = extract_feature_lengths(f'data/{data_file}')
    vizualize_data = data
    data_all = data
    if data_file == 'uniform.npy' :
        bins = np.linspace(np.min(data), np.max(data), 21)
        ax.hist(vizualize_data, bins=bins ,density=True)
    elif architecture == "escape_chain":
        bins = range(np.min(data), np.max(data))
        ax.hist(vizualize_data, bins=bins ,density=True)
    else:
        ax.hist(vizualize_data, bins='auto' ,density=True)
    likelyhoods, data_likelyhoods = mch_to_likelyhood_old(parameters , vizualize_data ,architecture, number_of_states)
    s_e = sh_entropy(data_all)
    matrix = generate_matrix(parameters, architecture, number_of_states)
    neg_log_likelyhood = obj_func_matrix(matrix, data_all, number_of_states)
    ax.plot(likelyhoods)
    ax.set_title(f'{number_of_states} states - cross entropy {neg_log_likelyhood:.2f}, dataset entorpy {s_e:.2f}')
    

def plot_best_training(architecture, mean_shift):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("outputs/statistics.tsv", sep='\t')
    grouped = df.groupby(["architecture", "input_file", "no_mean", 'fit'])
    #input_files = df["input_file"].unique()
    input_files = ['gaussian.npy', "far_gaussian.npy", "bimodal.npy", "uniform.npy"]
    #input_files = ['drosophila_melanogaster_genes_exons.bed','Gains_inc_chr.txt', 'H3K4me3_chr.txt', "hirt_chr.txt", "promoter_2k_chr_no_chrM.txt" ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for ax, input_file in zip(axs.flatten(), input_files):
        for name, group in grouped:
            arch, file, mean_sh, fit = name
            if file == input_file and arch == architecture and mean_sh == mean_shift:
                group = group.dropna(subset=['cross_entropy'])  # Drop NaN values in cross_entropy column if any
                min_cross_entropy_in_group = group["cross_entropy"].min()
                min_group_id = group.loc[group['cross_entropy'] == min_cross_entropy_in_group, 'order'].values[0]
                dir_path = os.path.join(outputs_dir, f'{min_group_id}')
                load_trained_model(ax, dir_path, file, fit)                

    
    
        
    plt.tight_layout()
    if mean_shift == 1:
        plt.savefig(f'pic_visualize/best_trainig_{architecture}.svg', format='svg')
    else :
        plt.savefig(f'pic_visualize/best_trainig_mean_sh_{architecture}.svg', format='svg')
    

# plot_best_training("chain", 1)
# plot_best_training("escape_chain", 1)
# plot_best_training("combined",1)
# plot_best_training("combined",0)

best_cross = pd.DataFrame(columns=["input_file", "number_of_states", "best_cross_entropy"])

best_cross = plot_cross_entropies("chain", 1, best_cross)
best_cross = plot_cross_entropies("escape_chain", 1, best_cross)
best_cross = plot_cross_entropies("combined", 1, best_cross)
best_cross = plot_cross_entropies("combined", 0, best_cross)


#plot_time("chain", 1)
#plot_mem_usage('chain', 1)

#plot_cross_entropies("escape_chain",1)


#plot_cross_entropies("combined",1)
#plot_best_training("combined", 1)
    