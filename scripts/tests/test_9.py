"""
Test the linear model
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from functions import complex_plot


# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']
pilotMatrix4N = np.float64(pilotMatrix4N)
configs = pilotMatrix4N[:, :N].T

# Assemble the vector of the samples h[k]
k = 5
hk = h_array[k, :N]

# Solve linear system
configs_inv = np.linalg.inv(configs)

# Get direct path
d = np.average(hk)

# Get nonlinear behaviour
div = 4
na = hk.reshape((div, -1))
nb = np.zeros(N // div, dtype=complex)
nb[:64] = np.average(na[1:, :], axis=0)[:64]
nb[64:] = np.average(na, axis=0)[64:]
n = np.tile(nb - d, div)
hl = np.zeros_like(hk)
hl[:64] = hk[:64] - n[:64] - d

# Get solution
c = configs_inv @ hl

# Test with the real linear coefficients
d_real = np.average((h_array[k, :N] + h_array[k, N: 2 * N]) / 2)
hl_real = (h_array[k, :N] - h_array[k, N:2 * N])[:64] / 2 + d_real
test_configs = pilotMatrix4N[:, :64]
hl_est = test_configs.T @ c + d
error = np.sum(np.abs(hl_real - hl_est) ** 2) / np.sum(np.abs(hl_real - np.average(hk[:64])) ** 2)
print(f"Error considering nonlinearities: {error}")

plt.figure()
plt.plot(np.abs(hl_est), 'r', label='estimated')
plt.plot(np.abs(hl_real), 'b', label='true')
plt.plot([0, len(hl_real) - 1], [np.abs(d), np.abs(d)], 'g', label='direct path')
plt.legend()

plt.figure()
plt.plot(np.angle(hl_est), 'r', label='estimated')
plt.plot(np.angle(hl_real), 'b', label='true')
plt.plot([0, len(hl_real) - 1], [np.angle(d), np.angle(d)], 'g', label='direct path')
plt.legend()


# Try to optimize
solution = (c * np.conj(d)).real
solution_truncated = np.sign(solution)

# Plot the vectors and the sum
plt.figure()
complex_plot(c.flatten(), 'ro-')
plt.title('unordered vectors')

plt.figure()
complex_plot(solution_truncated * c.flatten(), 'ro-')
complex_plot([solution_truncated @ c], 'b*-')
plt.title('ordered vectors and sum')

plt.figure()
plt.imshow(solution_truncated.reshape(64, 64))
