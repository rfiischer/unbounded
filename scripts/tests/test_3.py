"""
Try different model, with different gamma_1 and gamma_-1

It turns out not to be the right model
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

from functions import cost_function2


# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = 1  # data['M'][0, 0]
N = data['N'][0, 0]
thetas = data['pilotMatrix4N'].astype(np.float64)
mask1 = (thetas == 1).astype(np.float64)
mask2 = (thetas == -1).astype(np.float64)

# Assemble the vector of the samples h[3]
test = np.arange(0, N)
h0 = h_array[3, test].reshape((1, -1))

# Perform regression
t0 = np.concatenate(([1] * N, 0.0005 * np.random.randn(2 * (M + M * N))))
sol, nit, rc = fmin_tnc(lambda t: cost_function2(t, h0, mask1[:, test], mask2[:, test]),
                        t0)

# Get solution
ratios = sol[:N].reshape((-1, 1))
d = sol[N:N + M] + 1j * sol[N + M:N + 2 * M].reshape((-1, 1))
v = (sol[N + 2 * M:N + 2 * M + N * M] + 1j * sol[N + 2 * M + N * M:]).reshape((N, M))


# Test in remaining data
test_configs = ratios * mask1 - mask2
h_est = v.T @ test_configs + d
norm = np.sum(np.abs(h0 - np.average(h0)) ** 2)
error = np.sum(np.abs(h_array - h_est) ** 2) / norm

ex = 3
plt.plot(np.abs(h_est).T, 'ro', label='estimated')
plt.plot(np.abs(h_array[ex, :]), 'b*', label='true')
plt.show()
