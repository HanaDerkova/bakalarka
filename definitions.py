import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.stats import norm, entropy

# convention the escape state is always the last one
# should makesense in long run
# not gonna use softmax for params so far cause only 2 edges do not need it
# chain -> params the slucka pravdepodobnost on each state
# escape_chain -> params probability of escaping from each state
# combined -> self loop = 1, transition ,
def generate_matrix(parameters, architecture, number_of_states, k=None, l=None):
    if architecture == "chain":
        array = np.zeros((number_of_states,number_of_states))
        for i in range(number_of_states - 1):
            x = parameters[i]
            prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
            array[i][i] = prob_of_self_looping # set the diagonal whis is slucka to param[i]
            array[i][i + 1] = np.exp(0) / (np.exp(x) + np.exp(0))  # set the transition from i to i+1
        #array[number_of_states][number_of_states] = 0 # from escape state we aint gonna go anywhere
        matrix = array
    elif architecture == "escape_chain":
        matrix = np.zeros((number_of_states,number_of_states))
        for i in range(number_of_states - 1):
            x = parameters[i]
            prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
            matrix[i][number_of_states - 1] = prob_of_self_looping
            matrix[i][i + 1] = np.exp(0) / (np.exp(x) + np.exp(0))
    elif architecture == "combined":
        matrix = np.zeros((number_of_states,number_of_states))
        for i in range(number_of_states - 1):
            transition_prob_parameter = parameters[2*i]
            escaping_prob_parameter = parameters[2*i + 1]
            self_looping_parameter = 0
            sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            prob_escaping = np.exp(escaping_prob_parameter) / sum
            matrix[i][i] = prob_self_looping
            matrix[i][number_of_states - 1] = prob_escaping
            matrix[i][i + 1] = prob_trasnition
    elif architecture == "full":
        matrix = generate_full_mch(parameters=parameters, number_of_states=number_of_states)
    elif architecture == "cyclic":
        escape_state = number_of_states - 1
        matrix = np.zeros((number_of_states,number_of_states))
        for i in range(1, number_of_states - 2):
            x = parameters[i]
            prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
            matrix[i][i] = prob_of_self_looping # set the diagonal whis is slucka to param[i]
            matrix[i][i + 1] = np.exp(0) / (np.exp(x) + np.exp(0))
        #do first and last state separately: davaj dost pozor ako tam mas upratane tie parametre
        i = number_of_states - 3
        # transition_prob_parameter = parameters[0]
        # escaping_prob_parameter = parameters[i + 1]
        # self_looping_parameter = 0
        transition_prob_parameter = parameters[0]
        escaping_prob_parameter = 0
        #sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter)
        sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter)
        #prob_self_looping = np.exp(self_looping_parameter) / sum
        prob_trasnition = np.exp(transition_prob_parameter) / sum
        prob_escaping = np.exp(escaping_prob_parameter) / sum
        #matrix[0][0] = prob_self_looping
        matrix[0][1] = prob_trasnition
        matrix[0][escape_state] = prob_escaping
        # last state
        last_transient_st = number_of_states - 2
        #x = parameters[i + 2]
        x = parameters[i+1]
        prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
        matrix[last_transient_st][last_transient_st] = prob_of_self_looping # set the diagonal whis is slucka to param[i]
        matrix[last_transient_st][0] = np.exp(0) / (np.exp(x) + np.exp(0))
    elif architecture == "k-jumps":
       matrix = np.zeros((number_of_states,number_of_states))
       param_counter = 0
       for i in range(number_of_states - 1):
        if i % k != 0 or i == 0 or i < l :
            transition_prob_parameter = parameters[param_counter]
            escaping_prob_parameter = parameters[param_counter + 1]
            self_looping_parameter = 0
            sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            prob_escaping = np.exp(escaping_prob_parameter) / sum
            matrix[i][i] = prob_self_looping
            matrix[i][number_of_states - 1] = prob_escaping
            matrix[i][i + 1] = prob_trasnition
            param_counter += 2
        else:
            transition_prob_parameter = parameters[param_counter]
            escaping_prob_parameter = parameters[param_counter + 1]
            self_looping_parameter = 0
            back_loop_param = parameters[param_counter + 2]
            sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter) + np.exp(back_loop_param)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            prob_escaping = np.exp(escaping_prob_parameter) / sum
            prob_back_loop = np.exp(back_loop_param) / sum
            matrix[i][i] = prob_self_looping
            matrix[i][number_of_states - 1] = prob_escaping
            matrix[i][i + 1] = prob_trasnition
            matrix[i][i - l] = prob_back_loop
            param_counter += 3
       
    else:
        raise ValueError("Invalid architecture")

    return matrix

# using this in calculations (1 0 0 0 0 0 0) ,for the starting state
def create_vector(number_of_states):
    vec = np.zeros(number_of_states)  # Create an array of zeros
    vec[0] = 1  # Set the 0th element to 1
    return vec

# this has to be specific for architecture
def mch_to_likelyhood_old(parameters, data ,architecture, number_of_states, k=None, l=None):
    likelyhoods = [0]
    matrix = np.array( generate_matrix(parameters, architecture, number_of_states, k,l ) )
    #print(matrix)
    vector = np.array( create_vector(number_of_states) )
    result = vector[:] # we didd cpy here
    data_likelyhoods = []
    max_dist = max(data)
    for i in range(max_dist):
        result = np.dot(result, matrix) # zavinac
        likelyhoods.append(result[-1])  # [-1] gets u the last elem, likelyhoods[0] = 0
    for distance in data:
        data_likelyhoods.append(likelyhoods[distance])  # here ich mas pomiesane, nemalo by na tom zalezat, sum is komunitative
    return likelyhoods, data_likelyhoods

def precompute_powers(matrix, max_exponent):
    powers = [matrix]
    current_power = matrix
    for i in range(max_exponent.bit_length() - 1):
        current_power = np.dot(current_power, current_power)
        powers.append(current_power)
    return powers

def matrix_power(matrix, exponent, precomputed_powers):
    binary_exponent = bin(exponent)[2:][::-1]
    result = np.eye(matrix.shape[0])
    for i, bit in enumerate(binary_exponent):
        if bit == '1':
            result = np.dot(result, precomputed_powers[i])
    return result

# this has to be specific for architecture
def mch_to_likelyhood(parameters, data ,architecture, number_of_states, k=None, l=None):
    matrix = np.array( generate_matrix(parameters, architecture, number_of_states, k, l) )
    #print(matrix)
    vector = np.array( create_vector(number_of_states) )
    vector = vector[:] # we didd cpy here
    data_likelyhoods = []
    max_dist = max(data)
    precomputed_powers = precompute_powers(matrix, int(max_dist))
    unique_data_likelyhoods = {}
    for distance in np.unique(data):
      result = matrix_power(matrix, distance, precomputed_powers)
      prob_of_apsorption = (vector @ result)[-1]
      unique_data_likelyhoods[distance] = prob_of_apsorption
    for distance in data:
      # what herre???
      data_likelyhoods.append(unique_data_likelyhoods.get(distance))  # here ich mas pomiesane, nemalo by na tom zalezat, sum is komunitative
    return data_likelyhoods

# eps <- small constant to ensure that we do not log(0)
def objective_function(parameters, data ,architecture, number_of_states, k=None, l=None ,eps=1e-20):
    data_likelyhoods = mch_to_likelyhood(parameters, data ,architecture,number_of_states, k,l)
    negative_log_likelihood = -np.sum(np.log(np.array(data_likelyhoods) + eps)) / len(data)
    return negative_log_likelihood

def extract_feature_lengths(bed_file):
    feature_lengths = []  # Array to store feature lengths

    # Open the BED file
    with open(bed_file, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into columns based on tab delimiter
            columns = line.strip().split('\t')
            if len(columns) < 3:
              continue
            #print(columns)
            # Extract start and end positions from the appropriate columns
            start_position = int(columns[1])
            end_position = int(columns[2])

            # Calculate feature length and append to the array
            feature_length = end_position - start_position
            feature_lengths.append(feature_length)

    return feature_lengths

def create_parameters(input, number_of_states):
  vector = np.zeros((number_of_states -1) * 2)
  for i in range(number_of_states -1):
    vector[2*i] = input # transition
    vector[2*i + 1] = -5 #escaping je neparne probability, toto chceme nizke
  return vector

def compute(input, data, number_of_states, architecture):
  parameters = create_parameters(input, number_of_states)
  return objective_function(parameters, data ,architecture, number_of_states)

def expected_number_of_steps(t, Q):
  identity_matrix = np.eye(t)
  N = np.linalg.inv(identity_matrix - Q)
  column_vector = np.ones((t, 1))
  return (N @ column_vector)

def compute_mean(P):
  Q = P[:-1, :-1]
  t,z = Q.shape
  return expected_number_of_steps(t,Q)[0,0]

def variance_number_of_steps(P):
  Q = Q = P[:-1, :-1]
  t,z = Q.shape
  identity_t = np.eye(t)
  N = np.linalg.inv(identity_t - Q)
  exp_number_of_steps = expected_number_of_steps(t,Q)
  # Compute the Hadamard product of t with itself
  tsq = np.square(exp_number_of_steps)
  return (( 2 * N - identity_t) @ exp_number_of_steps - tsq)[0][0]

def objective_function_2(parameters, data, architecture, number_of_states, k=None, l=None):
  # Calculate the mean
  mean_data = np.mean(data)

  # Calculate the variance
  #variance_data = np.var(data)

  P = generate_matrix(parameters, architecture, number_of_states, k ,l)
  mean_phase_type = compute_mean(P)

  # # Calculate the variance
  # variance_phase_type = variance_number_of_steps(P)

  return (abs(mean_data - mean_phase_type))*100

def calculate_gap_lengths(bed_file):
    # Initialize variables
    last_end = 0
    last_chromosome = None
    gap_lengths = []

    # Open and read the BED file
    with open(bed_file, 'r') as f:
        for line in f:
            # Skip comment lines or empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            # Parse BED interval
            fields = line.strip().split('\t')
            chromosome = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            # Check if chromosome changes
            if chromosome != last_chromosome:
                last_end = 0
                last_chromosome = chromosome

            # Check for gap
            if start > last_end:
                gap_lengths.append(start - last_end)
            
            # Update last_end
            last_end = max(last_end, end)

    # Check for gap after the last annotation
    if last_end < end:
        gap_lengths.append(end - last_end)

    return gap_lengths

def check_rows_sum_to_one(matrix):
    row_sums = np.sum(matrix[:-1], axis=1)
    return np.allclose(row_sums, 1)

def generate_full_mch(parameters, number_of_states):
    matrix = np.zeros((number_of_states, number_of_states))
    counter = 0
    for row in range(number_of_states - 1):
        sum = np.exp(0)
        matrix_row = np.zeros(number_of_states)
        matrix_row[0] = np.exp(0)
        for column in range(1, number_of_states):
            parameter = np.exp(parameters[counter])
            sum += parameter
            matrix_row[column] = parameter
            counter += 1
        matrix[row] = matrix_row / sum
    if (check_rows_sum_to_one(matrix)) :
        return matrix
    else:
        print("smthing went wrong")

def optimize_once(args_list):
  np.random.seed()
  number_of_states, data, bounds, options, architecture, k, l = args_list
  if architecture == 'full':
      initial_guess = np.random.rand((number_of_states - 1) * (number_of_states -1))
  elif architecture == 'combined': 
      initial_guess = np.random.rand((number_of_states - 1) * 2)
  elif architecture == "chain" or architecture == 'escape_chain':
      initial_guess = np.random.rand(number_of_states - 1)
  elif architecture == 'cyclic' :
       initial_guess = np.random.rand(number_of_states -1) 
  elif architecture == "k-jumps" :
     initial_guess = np.random.rand((number_of_states - 1) * 2 + ((number_of_states -1 -l) // k))
  optimization_results = minimize(objective_function, initial_guess, method='L-BFGS-B', args=(data, architecture, number_of_states, k, l), bounds=bounds, options=options)
  return optimization_results

def opt_w_initialization(args_list, k=None, l=None):
  number_of_states, data, bounds, options, initial_guess, architecture = args_list
  optimization_results = minimize(objective_function, initial_guess, method='L-BFGS-B', args=(data, architecture, number_of_states, k,l), bounds=bounds, options=options)
  return optimization_results

def preprocessinig(args_list, k=None, l=None):
  np.random.seed()
  number_of_states, data, bounds, options, architecture = args_list
  if architecture == 'full':
      initial_guess = np.random.rand((number_of_states - 1) * (number_of_states - 1))
  elif architecture == 'combined': 
      initial_guess = np.random.rand((number_of_states - 1) * 2)
  elif architecture == "chain" or architecture == 'escape_chain':
      initial_guess = np.random.rand(number_of_states - 1)
  elif architecture == "k-jumps" :
     initial_guess = np.random.rand((number_of_states - 1) * 2 + ((number_of_states -1 -l) // k))
  optimization_results = minimize(objective_function_2, initial_guess, method='L-BFGS-B', args=(data, architecture, number_of_states), bounds=bounds, options=options)
  #print(optimization_results)
  return optimization_results

def sh_entropy(data):
    unique_values, value_counts = np.unique(data, return_counts=True)
    probabilities = value_counts / len(data)
    entropy_value = entropy(probabilities)
    return entropy_value

def max_mean_bounds(number_of_states, architecture, lower, upper):
    if architecture == "chain" or architecture == "escape_chain":
        params = np.full((number_of_states - 1), upper)
        P = generate_matrix(params, architecture, number_of_states)
    elif architecture == "combined":
        params = np.full((number_of_states - 1) * 2, lower)
        P = generate_matrix(params, architecture, number_of_states)
    elif architecture == "full":
        pass
    else:
        print("Unkonwn architecture")
    
    return compute_mean(P)

def pruning(matrix, threshold):
    new_matrix = np.copy(matrix)
    
    # Apply threshold condition for absolute values
    mask = np.abs(new_matrix) < threshold
    new_matrix[mask] = 0
    
    # Normalize each row to ensure the sum of weights in each row is 1
    row_sums = new_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    new_matrix /= row_sums[:, np.newaxis]

    return new_matrix

def matrix_to_likelyhood(matrix, data, number_of_states):
    likelyhoods = [0]
    vector = np.array( create_vector(number_of_states) )
    result = vector[:] # we didd cpy here
    data_likelyhoods = []
    max_dist = max(data)
    for i in range(max_dist):
        result = np.dot(result, matrix) # zavinac
        likelyhoods.append(result[-1])  # [-1] gets u the last elem, likelyhoods[0] = 0
    for distance in data:
        data_likelyhoods.append(likelyhoods[distance])  # here ich mas pomiesane, nemalo by na tom zalezat, sum is komunitative
    return likelyhoods, data_likelyhoods

def obj_func_matrix(matrix, data, number_of_states, eps=1e-20):
    lilelyhoods, data_likelyhoods = matrix_to_likelyhood(matrix, data,number_of_states)
    negative_log_likelihood = -np.sum(np.log(np.array(data_likelyhoods) + eps)) / len(data)
    return negative_log_likelihood


# number_of_states = 8
# k = 1
# l= 3
# parameters = np.random.rand( (number_of_states - 1) * 2 + ((number_of_states -1 -l) // k) )
# # # parameters = np.random.rand(number_of_states - 1)

# matrix = generate_matrix(parameters,"k-jumps", number_of_states, k ,l)

# output_str = np.array2string(matrix, separator=', ',suppress_small=True)
# output_str = output_str.replace('[', '').replace(']', '').replace('\n ', '\n')  # Remove brackets and space after newline

# print(output_str)
