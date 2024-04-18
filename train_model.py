import argparse
import matplotlib.pyplot as plt
from definitions import extract_feature_lengths, calculate_gap_lengths, preprocessinig, opt_w_initialization, mch_to_likelyhood_old, sh_entropy
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
import os
import sys

parser = argparse.ArgumentParser(description="parse arguments for training")
parser.add_argument('input_file', type=str, help='path to input bed file')
parser.add_argument('-o', dest='output_dir', type=str, help='path to output directory')
parser.add_argument('-architecture', type=str ,help='specify architecture for training')
parser.add_argument('-number_of_states','-ns', type=int, help='Number of states for Markov chain')
# parser.add_argument('--options','-opt', default=None , help='options for optimizer')
parser.add_argument('--threads', '-tr ',default=1, type=int, help='number of threads for optimization')
parser.add_argument('--fit', type=str, default='i', choices=['g', 'd', 'i'], help='fit gaps / intervals or load distribution from .npy file, defalut intervals')
parser.add_argument('--percent_visualize', '-p_v', default=100 ,type=int, help='percent of the data to be visualized, when hadling large bed file')
parser.add_argument('--sample', '-s', type=int, default=None, help='sample to preform training on, if bed file is too big to speed up training')

# Add optional parameters for L-BFGS-B optimization
parser.add_argument('--ftol', type=float, default=1e-5, help='tolerance for convergence')
parser.add_argument('--gtol', type=float, default=1e-5, help='gradient norm tolerance')
parser.add_argument('--max_iter', type=int, default=15000, help='maximum number of iterations')
parser.add_argument('--max_cor', type=int, default=10, help='maximum number of corrections')


args = parser.parse_args()

print(args)

if args.threads < 1:
    sys.exit("Error: Number of threads must be at least 1.")

if args.percent_visualize < 0:
    sys.exit("Error: Percent to visualize cannot be negative.")


if args.fit == 'g' :
    data = calculate_gap_lengths(args.input_file)
elif args.fit == 'i':
    data = extract_feature_lengths(args.input_file)
elif args.fit == "d":
    data = np.load(args.input_file)

#TODO : use the sample data for training then
if args.sample != None:
    sample_data = np.random.choice(data, size=args.sample, replace=False)


if args.architecture == "full" :
    bounds = [(-15, 15) for _ in range((args.number_of_states - 1) * (args.number_of_states))]
elif args.architecture == "combined":
    bounds = [(-15, 15) for _ in range((args.number_of_states - 1) * 2 )]
elif args.architecture == "chain" or args.architecture == "escape_chain" :
    bounds = [(-15, 15) for _ in range(args.number_of_states - 1)]

options = {'maxfun': 60000,
     'ftol': args.ftol,
     'gtol': args.gtol,
     'max_iter': args.max_iter,
     'max_cor': args.max_cor}  # Add optional parameters to the options dictionary


args_list = [(args.number_of_states, data, bounds, options, args.architecture )] * args.threads

with Pool(args.threads) as pool:
    # Perform optimization in parallel
    results = pool.map(preprocessinig, args_list)

#choose the most optimal initalization value
best_initalization = None
best_fun = float('inf')  # Initialize to positive infinity

# Iterate through all optimization results
for i, result in enumerate(results):
    # Check if the current result has a smaller fun value than the current best
    if result.fun < best_fun:
        best_initalization = result
        best_fun = result.fun
num_additional_initializations = args.threads - 1

# Sigma for Gaussian noise
sigma = 0.1

# Create additional initializations
initial_guesses = [best_initalization.x]  # Initialize with the best_initialization
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
    # Check if the current result has a smaller fun value than the current best
    if result.fun < best_fun:
        best_result = result
        best_fun = result.fun

# Print the best optimization result
if best_result:
    print(f"Best Optimization Result: {best_result}")
else:
    print("No optimization results found.")


vizualize_data = data 
if args.percent_visualize < 100:
    data_percentile = np.percentile(data, args.percent_visualize)
    vizualize_data = data[data <= data_percentile]

os.makedirs(args.output_dir, exist_ok=True)

plt.hist(vizualize_data, density=True)
likelyhoods, data_likelyhoods = mch_to_likelyhood_old(best_result.x , data ,args.architecture, args.number_of_states)
s_e = sh_entropy(data)
plt.plot(likelyhoods, label=f"states {args.number_of_states} cross entropy: {best_result.fun:.2f} sh entropy: {s_e:.2f}")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "data_vs_training.svg"))
plt.close()

parameters = best_result.x
cross_entropy = best_result.fun

# Save parameters to a text file
with open(os.path.join(args.output_dir, "trained_model.txt"), "a+") as file:
    file.write(f"Architecture: {args.architecture}\n")
    file.write(f"Number of States: {args.number_of_states}\n")
    file.write("Parameters:\n")
    for param in parameters:
        file.write(f"{param}\n")

np.save(os.path.join(args.output_dir, "cross_entropy.npy"), parameters)







