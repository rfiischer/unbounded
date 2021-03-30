"""
Test the linear model
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from functions import complex_plot


# Load data
data = loadmat("..\\datasets\\h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']

# # Choose N random configurations
# perm = np.random.permutation(4 * N)[:N]
# h_sel = h_array[:, perm]
# configs = pilotMatrix4N[:, perm]

# Choosing N random configurations unfortunately almost allways results in a singular matrix
pilotMatrix4N = np.float64(pilotMatrix4N)

# You can test wether the amplitude is different for +1 or -1
# It is sufficient to only change one value, since a common ratio can be factored in the coefficients
pilotMatrix4N[pilotMatrix4N == 1] = 1
configs = pilotMatrix4N[:, :N].T

# Assemble the vector of the samples h[k]
k = 5
h0 = h_array[k, :N]

# Solve linear system
configs_inv = np.linalg.inv(configs)

# Get solution
c = configs_inv @ h0

# Test in remaining data
test_configs = pilotMatrix4N
h0_est = test_configs.T @ c
h0_tru = h_array[k, :]
error = np.sum(np.abs(h0_tru - h0_est) ** 2) / np.sum(np.abs(h0_tru - np.average(h0)) ** 2)

plt.figure()
plt.plot(np.abs(h0_est), 'r', label='estimated')
plt.plot(np.abs(h0_tru), 'b', label='true')
plt.legend()

plt.figure()
plt.plot(np.angle(h0_est), 'r', label='estimated')
plt.plot(np.angle(h0_tru), 'b', label='true')
plt.legend()

# Try to optimize
c = c.reshape((-1, 1))
C = c @ np.conj(c.T)
R = np.real(C)

v, e = np.linalg.eig(R)
sums = c.T @ e

order = np.argsort(v)[::-1]
v = v[order]
e = e[:, order]

solution = e[:, 0]
solution_truncated = np.sign(solution)

# Plot the vectors and the sum
plt.figure()
complex_plot(c.flatten(), 'ro-')
plt.title('unordered vectors')

plt.figure()
complex_plot(solution_truncated * c.flatten(), 'ro-')
complex_plot(solution_truncated @ c, 'b*-')
plt.title('ordered vectors and sum')
