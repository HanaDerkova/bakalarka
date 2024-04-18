import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Define the parameters of the Gaussian distribution
mu = 20  # mean
sigma = 5 # standard deviation

num_samples = 500

samples = np.ceil(norm.ppf(np.random.rand(num_samples), mu, sigma)).astype(int)

np.save('data/gaussian.npy', samples)

axs[0, 0].hist(samples , bins='auto', density=True)
axs[0, 0].set_title('Gaussian')

#--------------------------------------------------------------------------------
# double gaussian
# Define the parameters of the Gaussian distribution
mu_1 = 15  # mean
sigma_1 = 3  # standard deviation

num_samples = 500

# Generate the heights using inverse transform sampling
samples_1 = np.ceil(norm.ppf(np.random.rand(num_samples), mu_1, sigma_1)).astype(int)

mu_2 = 40
sigma_2 = 5
samples_2 = np.ceil(norm.ppf(np.random.rand(num_samples), mu_2, sigma_2)).astype(int)

samples = np.concatenate([samples_1,samples_2])
np.save('data/bimodal.npy', samples)

axs[0, 1].hist(samples , bins='auto', density=True)
axs[0, 1].set_title('Bimodal')
#------------------------------------------------------------------------------------
# uniform
uniform = np.arange(1, 21)
np.save('data/uniform.npy', uniform)

axs[1, 0].hist(uniform,density=True)
axs[1, 0].set_title('Uniform')

#--------------------------------------------------------------------------------
# Define the parameters of the Gaussian distribution
mu = 350  # mean
sigma = 7 # standard deviation

num_samples = 500

samples = np.ceil(norm.ppf(np.random.rand(num_samples), mu, sigma)).astype(int)

np.save('data/far_gaussian.npy', samples)

axs[1, 1].hist(samples, bins='auto' ,density=True)
axs[1, 1].set_title('Gaussian w far mean')

plt.savefig('distributions_plot.png')
#_____________________________________________________________________________________