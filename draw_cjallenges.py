import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

gaussian = np.load("data/gaussian.npy")
far_gaussian = np.load('data/far_gaussian.npy')
bimodal = np.load("data/bimodal.npy")
uniform = np.load('data/uniform.npy')

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for ax, data, title in zip(axs.flatten(), [gaussian, far_gaussian, bimodal, uniform], ['Gaussian small samples', 'Gaussian big samples', 'Bimodal', 'Uniform']):
    if title == 'Uniform':
        bins = np.linspace(np.min(data), np.max(data), 21)
        ax.hist(data, bins=bins, density=True)
    else:
        ax.hist(data, bins='auto', density=True)
    ax.set_title(title)
    if title != "Gaussian big samples":
        ax.set_xlim(0, 60)  # Set the same x-axis limits for all subplots
    else :
        ax.set_xlim(200, None) 
    ax.set_ylim(0, 0.09)  # Set the same y-axis limits for all subplots

plt.tight_layout()
plt.savefig(f'pic_visualize/challenges.svg', format='svg')
