"""
Try different model, with different gamma_1 and gamma_-1

It turns out not to be the right model
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

from functions import cost_function4


# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
thetas = data['pilotMatrix4N'].astype(np.float64)
k = 2
hk = h_array[k, :]


def test_model(size):

    # Assemble the vector of the samples h[k]
    test = np.arange(0, size)
    factor = np.max(np.abs(hk[test]))
    h0 = hk[test] / factor

    # Perform regression
    t0 = 0.0005 * np.random.randn(2 * N + 2)
    sol, nit, rc = fmin_tnc(lambda t: cost_function4(t, h0, thetas[:, test]),
                            t0)

    # Get solution
    c = (sol[0:N] + 1j * sol[N:2 * N]) * factor
    d = (sol[2 * N] + 1j * sol[2 * N + 1]) * factor

    # Test in remaining data
    test_configs = thetas
    hk_est = test_configs.T @ c + d
    norm = np.sum(np.abs(hk - np.average(hk)) ** 2)
    error = np.sum(np.abs(hk - hk_est) ** 2) / norm

    return c, d, error, hk_est


c1, d1, e1, _ = test_model(N)
c2, d2, e2, _ = test_model(2 * N)
c3, d3, e3, est = test_model(4 * N)

print(f"Error with N: {e1}\n"
      f"Error with 2N: {e2}\n"
      f"Error with 4N: {e3}\n")

plt.plot(np.abs(hk), 'b', label='Target')
plt.plot(np.abs(est), 'r', label='Linear Regression')
plt.legend()
plt.xlabel(r"Configuration Index $\nu$")
plt.ylabel(fr"$|h_\theta^\nu[{k}]|$")
plt.show()


# Estimate linear component
hm = (hk[:N] - hk[N:2 * N]) / 2
c4 = thetas[:, :N] @ hm / N
d4 = np.average(hk[:N] - hm)
hk_est = thetas.T @ c4 + d4
norm = np.sum(np.abs(hk - np.average(hk)) ** 2)
e4 = np.sum(np.abs(hk - hk_est) ** 2) / norm
print(f"Error using the linear component: {e4}")

plt.figure(figsize=(5, 5))
plt.imshow(np.abs(c3).reshape(64, 64))
plt.title(f"|$C_n[{k}]|$")
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
