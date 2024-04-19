import pandas as pd
import json

configfile: "config_folder/config.yaml"


output_dir = config.get("output_directory", "output")
experiments_file = config["experiments_file"]
data_dir = config["data_file"]
experiments = pd.read_csv(experiments_file, sep=",")


rule train_model:
   input: inputt=lambda w: f"{data_dir}/" + experiments.loc[experiments["order"] == int(w.id), "input_file"].values[0]
   params:
       architecture = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "architecture"].values[0],
       number_of_states = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "number_of_states"].values[0],
       #params_json = lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0]),
       #number_of_tries = lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0])["number_of_tries"],
       #ftol = lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0])["ftol"],
       #maxiter = lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0])["maxiter"],
       #max_cor = lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0])["max_cor"],
       #fit = lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0])["fit"]
        fit = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "fit"].values[0]
   
   
#    threads: lambda wildcards: json.loads(experiments.loc[experiments["id"] == wildcards.id, "params_json"].values[0])["threads"]

   output:
       model=f"{output_dir}/{{id}}/trained_model.txt",
       cross_entropy=f"{output_dir}/{{id}}/cross_entropy.csv",
       histogram_viz=f"{output_dir}/{{id}}/data_vs_training.svg",
       metrics=f"{output_dir}/{{id}}/metrics.txt"

   shell: f"""(/usr/bin/time -f "%e %M" python3 train_model.py {{input}} \
        -o {output_dir}/{{wildcards.id}} \
        -architecture {{params.architecture}} \
        -ns {{params.number_of_states}}  \
        --fit '{{params.fit}}' """
        f""") 2>&1 | tail -n1 > {{output.metrics}}"""


rule collect_statistics:
   input:
       cross_entropies=expand(f"{output_dir}/{{id}}/cross_entropy.csv", id=experiments["order"].values),
       experiments_list=experiments_file
   output: f"{output_dir}/statistics.tsv"
   run:
       cross_entropies = pd.concat([pd.read_csv(f, sep=",") for f in input.cross_entropies])
       print(cross_entropies)
       experiments = pd.read_csv(input.experiments_list, sep=",")
       print(experiments)
       cross_entropies = cross_entropies.merge(experiments, on="order")
       cross_entropies.to_csv(output[0], sep="\t", index=False)
      
      
      
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--inputs', nargs='+', help='List of input arguments')
# args = parser.parse_args()
#
# input_list = args.inputs
# print("Input list:", input_list)

rule all:
   #default_target: True
   input: f"{output_dir}/statistics.tsv" , expand(f"{output_dir}/{{id}}/trained_model.txt", id=experiments["order"].values),





# import csv

# # Read configuration from CSV file
# config = {}
# with open("config.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row  in reader:
#         setup_name = str(row["order"])  # Get the setup name from the "order" column
#         config[setup_name] = {
#             "input_file": row["input file"],
#             "output_file": row["output file"],
#             "architecture": row["architecture"],
#             "number_of_states": row["number of states"],
#             "threads": row["threads"],
#             "fit": row["fit"]
#         }
# print(config.keys())
# #print(config[1]["input_file"])

# # Define targets and rules
# rule all:
#     input:
#         expand("{setup}_output/data_vs_training.svg", setup=config.keys()),
#         expand("{setup}_output/trained_model.txt", setup=config.keys()),
#         expand("{setup}_output/cross_entropy.npy", setup=config.keys())

# rule train_model:
#     input:
#         input_file="data/" + config["2"]["input_file"]
#     output:
#         # directory("{setup}_output"),
#         "{setup}_output/data_vs_training.svg",
#         "{setup}_output/cross_entropy.npy",
#         "{setup}_output/trained_model.txt"
#     params:
#          architecture=lambda setup: config["{setup}"]["architecture"],
#          number_of_states=lambda setup: config["{setup}"]["number_of_states"],
#          threads=lambda setup: config["{setup}"]["threads"],
#          fit=lambda setup: config["{setup}"]["fit"]

#     shell:
#         """
#         echo 'h'
#         """

#     # shell:
#         # """
#         # python3 train_model.py {input} \
#         # -o {output}/ \
#         # -architecture {params.architecture} \
#         # -ns {params.number_of_states} \
#         # --fit {params.fit}
#         # """

# #TODO : u still do not have threads here !!!! add them when running on the cluster