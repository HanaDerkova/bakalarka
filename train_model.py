import argparse
import matplotlib.pyplot as plt
from definitions import obj_func_matrix, generate_matrix ,optimize_once, extract_feature_lengths, calculate_gap_lengths, preprocessinig, opt_w_initialization, mch_to_likelyhood_old, sh_entropy
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
import os
import pandas as pd
import sys
import random
import logging

parser = argparse.ArgumentParser(description="parse arguments for training")
#MANDATORY ARGUMENTS------------------------------------------------------------------------------
parser.add_argument('input_file', type=str, help='path to input bed file')
parser.add_argument('-o', dest='output_dir', type=str, help='path to output directory')
parser.add_argument('-architecture', type=str ,help='specify architecture for training')
parser.add_argument('-number_of_states','-ns', type=int, help='Number of states for Markov chain')
parser.add_argument('-log_f', type=str, default='trainig_log' ,help="path to logging file")
parser.add_argument('--no_mean', type=bool, default=False, help='Turn of mean shifting pre trainig' )
#ARGUMENTS FOR THREADS AND GAPS/INTERVAL/DISTIRBUTION FITTING----------------------------------
parser.add_argument('--threads', '-tr ',default=1, type=int, help='number of threads for optimization')
parser.add_argument('--fit', type=str, default='i', choices=['g', 'd', 'i'], help='fit gaps / intervals or load distribution from .npy file, defalut intervals')
#TRAINING OPTIONS ARGUMENTS------------------------------------------------------------------------------------------
parser.add_argument("--training_opt", nargs='+', type=int, help="number_of_tires, sample_size, percent_to_vizualize")
#OPTIMIZER ARGUMENTS--------------------------------------------------------------------------------------------------------
parser.add_argument('--opt_options', nargs='+', type=float, help='List of input arguments: ftol gtol maxiter maxcor, mind that maxiter and maxcor must be integers')

args = parser.parse_args()

log_file = os.path.join(args.output_dir,args.log_f )
logging.basicConfig(filename=log_file, level=logging.INFO)

logging.info(args)

# setting defalut values for training optiions arguments
number_of_tries = 1
sample_size = None
percent_visualize = 100

# steeting trainig options arguments if they were given
if args.training_opt is not None:
    if len(args.training_opt) != 3:
        sys.exit("Error : provide exactly 3 values in order number_of_tires, sample_size, percent_to_vizualize") 
    number_of_tries, sample_size, percent_visualize = args.training_opt

    if percent_visualize < 0:
        sys.exit("Error: Percent to visualize cannot be negative.")
    if sample_size < 0:
        sys.exit("Error: Sample size cannot be negative.")
    if number_of_tries < 0:
        sys.exit("Error: NUmber of tries cannot be negative.")

# chcking threads 
if args.threads < 1:
    sys.exit("Error: Number of threads must be at least 1.")

# loading into data the right form of distibution
if args.fit == 'g' :
    print("Ensure that your bed file is already ordered!")
    data = calculate_gap_lengths(args.input_file)
elif args.fit == 'i':
    data = extract_feature_lengths(args.input_file)
elif args.fit == "d":
    data = np.load(args.input_file)

# setting bounds for given architecture
if args.architecture == "full" :
    bounds = [(-15, 15) for _ in range((args.number_of_states - 1) * (args.number_of_states))]
elif args.architecture == "combined":
    bounds = [(-15, 15) for _ in range((args.number_of_states - 1) * 2 )]
elif args.architecture == "chain" or args.architecture == "escape_chain" :
    bounds = [(-15, 15) for _ in range(args.number_of_states - 1)]

# default options for options
options = {'maxfun': 60000}

# setting optimizer options 
if args.opt_options is not None:
    if len(args.opt_options) != 4:
        sys.exit("Error : provide exactly four input values: ftol gtol max_iter max_cor")
    ftol, gtol, max_iter, max_cor = args.opt_options
    options = {'maxfun': 60000,
     'ftol': ftol,
     'gtol': gtol,
     'maxiter': int(max_iter),
     'maxcor': int(max_cor)}


overall_best_result = None
overall_best_fun = float('inf')

# storing data here
data_all = data.copy()

if not args.no_mean:
    for iter in range(number_of_tries): 

        # sample data for training in sample size was specfied
        if sample_size != None :
            sample = random.choices(data_all, k=sample_size)
            data = sample
        
        # create args list
        args_list = [(args.number_of_states, data, bounds, options, args.architecture )] * args.threads

        # mean shifting
        with Pool(args.threads) as pool:
            # Perform optimization in parallel
            results = pool.map(preprocessinig, args_list)

        #choose the most optimal initalization value
        best_initalization = None
        best_fun = float('inf')

        # Iterate through all optimization results
        for i, result in enumerate(results):
            # Check if the current result has a smaller fun value than the current best
            if result.fun < best_fun:
                best_initalization = result
                best_fun = result.fun
        num_additional_initializations = args.threads - 1

        # Sigma for Gaussian noise
        sigma = 0.1

        # Create additional initializations for next level of trainig
        initial_guesses = [best_initalization.x]
        for i in range(num_additional_initializations):
            # Generate Gaussian noise
            noise = np.random.normal(loc=0, scale=sigma, size=len(best_initalization.x))

            # Add noise to the best_initialization
            initialization = best_initalization.x + noise

            # Append the new initialization to the list
            initial_guesses.append(initialization)

        args_list = [(args.number_of_states, data, bounds, options, i , args.architecture) for i in initial_guesses]

        with Pool(args.threads) as pool:
            results = pool.map(opt_w_initialization, args_list)
            
        best_result = None
        best_fun = float('inf')  # Initialize to positive infinity

        # Iterate through all optimization results
        for i, result in enumerate(results):
            if result.fun < best_fun:
                best_result = result
                best_fun = result.fun

        # Print the best optimization result
        if best_result:
            logging.info(f"Best Optimization Result for iteration {iter} : {best_result}")
        else:
            print("No optimization results found.")

        if best_fun < overall_best_fun:
            overall_best_fun = best_fun
            overall_best_result = best_result
else :
    for iter in range(number_of_tries): 

        # sample data for training in sample size was specfied
        if sample_size != None :
            sample = random.choices(data_all, k=sample_size)
            data = sample

        #number_of_states, data, bounds, options, architecture
        args_list = [(args.number_of_states, data, bounds, options, args.architecture) ] * args.threads

        with Pool(args.threads) as pool:
            results = pool.map(optimize_once, args_list)
            
        best_result = None
        best_fun = float('inf')  # Initialize to positive infinity

        # Iterate through all optimization results
        for i, result in enumerate(results):
            if result.fun < best_fun:
                best_result = result
                best_fun = result.fun

        # Print the best optimization result
        if best_result:
            logging.info(f"Best Optimization Result for iteration {iter} : {best_result}")
        else:
            logging.error("No optimization results found.")

        if best_fun < overall_best_fun:
            overall_best_fun = best_fun
            overall_best_result = best_result

vizualize_data = data_all 
if percent_visualize < 100:
    data_percentile = np.percentile(data_all, percent_visualize)
    vizualize_data = data_all[data_all <= data_percentile]


os.makedirs(args.output_dir, exist_ok=True)

plt.hist(vizualize_data, density=True)
likelyhoods, data_likelyhoods = mch_to_likelyhood_old(overall_best_result.x , vizualize_data ,args.architecture, args.number_of_states)
s_e = sh_entropy(data_all)
matrix = generate_matrix(overall_best_result.x, args.architecture, args.number_of_states)
neg_log_likelyhood = obj_func_matrix(matrix, data, args.number_of_states)
plt.plot(likelyhoods, label=f"states {args.number_of_states} cross entropy: {neg_log_likelyhood:.2f} sh entropy: {s_e:.2f}")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "data_vs_training.svg"))
plt.close()

parameters = overall_best_result.x
cross_entropy = overall_best_result.fun

# Save parameters to a text file
with open(os.path.join(args.output_dir, "trained_model.txt"), "w") as file:
    file.write(f"Architecture: {args.architecture}\n")
    file.write(f"Number of States: {args.number_of_states}\n")
    file.write("Parameters:\n")
    for param in parameters:
        file.write(f"{param}\n")


order = order = os.path.basename(args.output_dir)

# Create a dictionary with column names and values
data = {
    "order": [order],
    "cross_entropy": [cross_entropy]  # Extract the single value from the array
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(args.output_dir, "cross_entropy.csv"), index=False)

logging.shutdown()






