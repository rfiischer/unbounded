import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

from functions import complex_plot
from functions import cost_function1


# Load data
data = loadmat("..\\datasets\\h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
thetas = data['pilotMatrix4N'].astype(np.float64)
mask1 = (thetas == 1).astype(np.float64)
mask2 = (thetas == -1).astype(np.float64)

# Assemble the vector of the samples h[3]
test_len = 2 * N
h0 = h_array[3, :test_len]

# Perform regression
norm = np.sum(np.abs(h0 - np.average(h0)) ** 2)
t0 = np.concatenate(([1.5], 0.0005 * np.random.randn(4096 * 2)))
sol, nit, rc = fmin_tnc(lambda t: cost_function1(t, h0, mask1[:, :test_len], mask2[:, :test_len]),
                        t0)

# Get solution
a = sol[0]
c = sol[1:N + 1] + 1j * sol[N + 1:]

# Test in remaining data
test_configs = a * mask1 - mask2
h0_est = test_configs.T @ c
h0_tru = h_array[3, :]
error = np.sum(np.abs(h0_tru - h0_est) ** 2) / norm

plt.plot(np.abs(h0_est), 'ro', label='estimated')
plt.plot(np.abs(h0_tru), 'b*', label='true')
plt.show()
