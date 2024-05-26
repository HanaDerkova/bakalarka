import pandas as pd
import json

configfile: "config_folder/config.yaml"


output_dir = config.get("output_directory", "output")
experiments_file = config["experiments_file"]
data_dir = config["data_file"]
experiments = pd.read_csv(experiments_file, sep=",")
default_target = f"{output_dir}/statistics.tsv"
num_of_threads = config["threads"]


def f(wildcards):
    print(type( experiments.loc[experiments["order"] == int(wildcards.id), "threads"].values[0]))
    return experiments.loc[experiments["order"] == int(wildcards.id), "threads"].values[0]

rule train_model:
   input: 
        inputt=lambda w: f"{data_dir}/" + experiments.loc[experiments["order"] == int(w.id), "input_file"].values[0]
   params:
       architecture = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "architecture"].values[0],
       number_of_states = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "number_of_states"].values[0],
       number_of_tries = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "train_opt_json"].values[0])["number_of_tries"],
       sample_size = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "train_opt_json"].values[0])["sample_size"],
       percent_to_visualize = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "train_opt_json"].values[0])["percent_to_visualize"],
       ftol = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "params_json"].values[0])["ftol"],
       gtol = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "params_json"].values[0])["gtol"],
       maxiter = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "params_json"].values[0])["maxiter"],
       max_cor = lambda wildcards: json.loads(experiments.loc[experiments["order"] == int(wildcards.id), "params_json"].values[0])["maxcor"],
       fit = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "fit"].values[0],
       no_mean = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "no_mean"].values[0],
       thr = lambda wildcards: experiments.loc[experiments["order"] == int(wildcards.id), "threads"].values[0],
       k = lambda wildcards: int(experiments.loc[experiments["order"] == int(wildcards.id), "k"].values[0]) if pd.notna(experiments.loc[experiments["order"] == int(wildcards.id), "k"].values[0]) else None,
       l = lambda wildcards: int(experiments.loc[experiments["order"] == int(wildcards.id), "l"].values[0]) if pd.notna(experiments.loc[experiments["order"] == int(wildcards.id), "l"].values[0]) else None
   threads:
         num_of_threads
   output:
       model=f"{output_dir}/{{id}}/trained_model.txt",
       cross_entropy=f"{output_dir}/{{id}}/cross_entropy.csv",
       histogram_viz=f"{output_dir}/{{id}}/data_vs_training.svg",
       metrics=f"{output_dir}/{{id}}/metrics.txt"

   shell: 
        f"""(/usr/bin/time -f "%e %M" python3 train_model.py {{input}} \
        -o {output_dir}/{{wildcards.id}} \
        -architecture {{params.architecture}} \
        -ns {{params.number_of_states}}  \
        --fit '{{params.fit}}' \
        --opt_options {{params.ftol}} {{params.gtol}} {{params.maxiter}} {{params.max_cor}} \
        --no_mean {{params.no_mean}} \
        --threads {{threads}}  \
        --training_opt {{params.number_of_tries}} {{params.sample_size}} {{params.percent_to_visualize}} \
        --k {{params.k}} --l {{params.l}}  """
        f""") 2>&1 | tail -n1 > {{output.metrics}}"""

rule time_mem_to_csv:
    input : inputt=lambda w: f"{output_dir}/{w.id}/"  + 'metrics.txt'
    output :
        metrics=f"{output_dir}/{{id}}/metrics.csv",
    shell :
        f"""python3 time_mem_to_csv.py {output_dir}/{{wildcards.id}} -o {{wildcards.id}} """



#TODO: check wather they concatenate correclty 
rule collect_statistics:
   input:
       cross_entropies=expand(f"{output_dir}/{{id}}/cross_entropy.csv", id=experiments["order"].values),
       metrics=expand(f"{output_dir}/{{id}}/metrics.csv", id=experiments["order"].values),
       experiments_list=experiments_file
   output: f"{output_dir}/statistics.tsv"
   run:
       cross_entropies = pd.concat([pd.read_csv(f, sep=",") for f in input.cross_entropies])
       final_metrics = pd.concat([pd.read_csv(f, sep=",") for f in input.metrics])
       experiments = pd.read_csv(input.experiments_list, sep=",")
       cross_entropies = cross_entropies.merge(experiments, on="order")
       final_metrics =final_metrics.merge(cross_entropies, on="order")
       final_metrics.to_csv(output[0], sep="\t", index=False)
      
      
rule all:
   input: f"{output_dir}/statistics.tsv"
   #, expand(f"{output_dir}/{{id}}/trained_model.txt", id=experiments["order"].values),


