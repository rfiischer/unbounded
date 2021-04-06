"""
Try different model, with different gamma_1 and gamma_-1

It turns out not to be the right model
"""

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc

from functions import cost_function3


# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
thetas = data['pilotMatrix4N'].astype(np.float64)

# Assemble the vector of the samples h[3]
test = np.arange(0, 3 * N)
h0 = h_array[3, test] / np.max(np.abs(h_array[3, test]))

# Perform regression
guess = thetas[:, :N] @ h0[:N] / N
t0 = np.concatenate((guess.real, guess.imag, [0, 0]))
sol, nit, rc = fmin_tnc(lambda t: cost_function3(t, h0, N, thetas[:, test]),
                        t0, ftol=1e-20, xtol=1e-20, pgtol=1e-20)

# Get solution
c = (sol[:N] + 1j * sol[N:2 * N]) * np.max(np.abs(h_array[3, test]))
d = (sol[2 * N] + 1j * sol[2 * N + 1]) * np.max(np.abs(h_array[3, test]))

# Test in remaining data
test_configs = thetas
h0_est = test_configs.T @ c + d
h0_tru = h_array[3, :]
norm = np.sum(np.abs(h0 - np.average(h0)) ** 2)
error = np.sum(np.abs(h0_tru - h0_est) ** 2) / norm

plt.plot(np.abs(h0_est), 'r', label='estimated')
plt.plot(np.abs(h0_tru), 'b', label='true')
plt.show()
