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
hk = h_array[k, :N]

# Solve linear system
configs_inv = np.linalg.inv(configs)

# Get direct path
d = np.average(h_array[k, :N] + h_array[k, N:2 * N] + h_array[k, 2 * N:3 * N] + h_array[k, 3 * N:]) / 4

# Get solution
c = configs_inv @ (hk - d)

# Test in remaining data
test_configs = pilotMatrix4N
hk_est = test_configs.T @ c + d
hk_tru = h_array[k, :]
error = np.sum(np.abs(hk_tru - hk_est) ** 2) / np.sum(np.abs(hk_tru - np.average(hk)) ** 2)
print(f"Error with best estimate of direct path: {error}")

plt.figure()
plt.plot(np.abs(hk_est), 'r', label='estimated')
plt.plot(np.abs(hk_tru), 'b', label='true')
plt.plot([0, len(hk_est) - 1], [np.abs(d), np.abs(d)], 'g', label='direct path')
plt.legend()

plt.figure()
plt.plot(np.angle(hk_est), 'r', label='estimated')
plt.plot(np.angle(hk_tru), 'b', label='true')
plt.plot([0, len(hk_est) - 1], [np.angle(d), np.angle(d)], 'g', label='direct path')
plt.legend()


# Get direct path
d = np.average(h_array[k, :N])

# Get solution
c = configs_inv @ (hk - d)

# Test in remaining data
test_configs = pilotMatrix4N
hk_est = test_configs.T @ c + d
hk_tru = h_array[k, :]
error = np.sum(np.abs(hk_tru - hk_est) ** 2) / np.sum(np.abs(hk_tru - np.average(hk)) ** 2)
print(f"Error with conservative estimate of direct path: {error}")

plt.figure()
plt.plot(np.abs(hk_est), 'r', label='estimated')
plt.plot(np.abs(hk_tru), 'b', label='true')
plt.plot([0, len(hk_est) - 1], [np.abs(d), np.abs(d)], 'g', label='direct path')
plt.legend()

plt.figure()
plt.plot(np.angle(hk_est), 'r', label='estimated')
plt.plot(np.angle(hk_tru), 'b', label='true')
plt.plot([0, len(hk_est) - 1], [np.angle(d), np.angle(d)], 'g', label='direct path')
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
