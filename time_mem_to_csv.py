import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="load input output dir")

parser.add_argument('input_dir', type=str, help='path to input bed file')
parser.add_argument('-order','-o', type=int, help='specify order')

args = parser.parse_args()

print(args)

with open(os.path.join(args.input_dir, "metrics.txt"), 'r') as file:
        # Read the line and split it into two numbers
    numbers = file.readline().strip().split()

    # Ensure there are exactly two numbers
    if len(numbers) != 2:
        print("Error: The file does not contain two numbers on the same line.")
    else:
        # Convert the numbers to floats
        try:
            time = float(numbers[0])
            memory = float(numbers[1])
            data = {
                'order' : [args.order],
                'time' : [time],
                'memory' : [memory]
            }

            df = pd.DataFrame(data)

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(args.input_dir, "metrics.csv"), index=False)

            print("First number:", time)
            print("Second number:", memory)
        except ValueError:
            print("Error: The file contains non-numeric values.")