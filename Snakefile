import csv

# Read configuration from CSV file
config = {}
with open("config.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row  in reader:
        setup_name = row["order"]  # Get the setup name from the "order" column
        config[setup_name] = {
            "input_file": row["input file"],
            "output_file": row["output file"],
            "architecture": row["architecture"],
            "number_of_states": row["number of states"],
            "threads": row["threads"],
            "fit": row["fit"]
        }
print(config.keys())
#print(config[1]["input_file"])

# Define targets and rules
rule all:
    input:
        expand("{setup}_output/data_vs_training.svg", setup=config.keys()),
        expand("{setup}_output/trained_model.txt", setup=config.keys()),
        expand("{setup}_output/cross_entropy.npy", setup=config.keys())

rule train_model:
    input:
        input_file="data/" + config["2"]["input_file"]
    output:
        directory("{setup}_output"),
        "{setup}_output/data_vs_training.svg",
        "{setup}_output/cross_entropy.npy",
        "{setup}_output/trained_model.txt"
    params:
        architecture=lambda setup: config["{setup}"]["architecture"],
        number_of_states=lambda setup: config["{setup}"]["number_of_states"],
        threads=lambda setup: config["{setup}"]["threads"],
        fit=lambda setup: config["{setup}"]["fit"]

    shell:
    """
    echo 'h'
    """

    # shell:
        # """
        # python3 train_model.py {input} \
        # -o {output}/ \
        # -architecture {params.architecture} \
        # -ns {params.number_of_states} \
        # --fit {params.fit}
        # """

#TODO : u still do not have threads here !!!! add them when running on the cluster