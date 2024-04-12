import argparse

parser = argparse.ArgumentParser(description="parse arguments for training")
parser.add_argument('input_file', type=str, help='path to input bed file')
parser.add_argument('-o', dest='output_dir', type=str, help='path to output directory')
parser.add_argument('-architecture', type=str ,help='specify architecture for training')
parser.add_argument('--number_of_states','-ns' type=int, help='Number of states for Markov chain')
parser.add_argument('--opt', default=None , help='options for optimizer')
parser.add_argument('--threads', '-tr ',default=1, type=int, help='number of threads for optimization')

args = parser.parse_args()

print(args)
