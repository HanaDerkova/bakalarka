# Software tool for moddeling gennomic annotations using Absobing Markov chains

## Installation

```shell
install miniconda3
conda install mamba
mamba create -n training_env python=3.11 numpy scipy matplotlib pandas 
conda activate training_env
```

## Demo

```shell
conda activate training_env
python3 train_model.py data/gaussian.npy -o outputs -architecture combined -ns 10 --fit 'd' --opt_options 1e-06 1e-06 15000 10 --no_mean 0 --threads 1 --training_opt 10 500 100
```

## Usage

```
usage: train_model.py [-h] [-o OUTPUT_DIR] [-architecture ARCHITECTURE]
                      [-number_of_states NUMBER_OF_STATES] [-log_f LOG_F] [--no_mean NO_MEAN]
                      [--threads THREADS] [--fit {g,d,i}] [--training_opt TRAINING_OPT [TRAINING_OPT ...]]
                      [--opt_options OPT_OPTIONS [OPT_OPTIONS ...]] [--k K] [--l L]
                      input_file

Software tool for modeling genomic annotations: trains an Absorbing Markov Chain to model the distribution of annotation lengths from a given file.

The input file should be tab-separated with three columns:

    1. Chromosome name
    2. Start of an interval
    3. End of an interval

For fitting gap lengths, the input file should be sorted. Otherwise, we recommend using Bedtools complement and then running the tool for modeling interval lengths.

The tool outputs four files:

    1. trained_model.txt    Specifies the architecture, number of states, and trained parameter weights.
    2. data_vs_training.svg Displays the learned phase-type distribution and annotation distribution.
    3. cross_entropy.csv    Contains the resulting cross-entropy values.
    4. training_log         Logs the training process.

positional arguments:
  input_file            Path to input bed file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR         Path to output directory.
  -architecture ARCHITECTURE
                        Specify architecture for training. Available architectures: chain, escape_chain,
                        cyclic, full, k-jumps
  -number_of_states NUMBER_OF_STATES, -ns NUMBER_OF_STATES
                        Number of states for Markov chain.
  -log_f LOG_F          Path to logging file.
  --no_mean NO_MEAN     Turn of mean shifting pre-trainig. Defaultly set to pre-train. Options [0-False,
                        1-True]
  --threads THREADS, -tr  THREADS
                        Number of threads for optimization.
  --fit {g,d,i}         Fit gaps / intervals / load distribution from .npy file. Default is intervals.
  --training_opt TRAINING_OPT [TRAINING_OPT ...]
                        Options for trainig: number_of_tires (for optimization), sample_size (for sub-
                        sampling in case of bis sample size), percent_to_vizualize (for visualizing the
                        final fit). Mind that arguments have to be present in this order.
  --opt_options OPT_OPTIONS [OPT_OPTIONS ...]
                        List of input arguments for L-BFGS-B optimizer method: ftol, gtol, maxiter, maxcor.
                        Mind that maxiter and maxcor must be integers and arguments have to be presentet in
                        this order.
  --k K                 k parameter for k-jumps architecture, specifing the frequency of a jump (in our
                        experiments this was defaultly set to 1)
  --l L                 l parameter for k-jumps architecture, specifying the lenght of a jump.
```

## Using Snakemake

Snakemake is a workflow management system that enables reproducible and scalable data analyses.

In this project, Snakemake is used to automate and manage the training of the Absorbing Markov Chain models on genomic annotation data.

## Usage

The Snakemake workflow for this project is defined in the Snakefile provided in the repository. To run the experiments defined in config_folder/expetiments.csv and collect staistic about these expetionemst in statistics.tsv, use the following command:

## Demo

```shell
snakemake all
```