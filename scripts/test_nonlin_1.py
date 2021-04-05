import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt

from functions import cost_function4, compute_features


# Load data
data = loadmat("..\\datasets\\h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']


# Get nonlinear features
pilotMatrix4N = np.float64(pilotMatrix4N)
p1 = pilotMatrix4N[:, :N]
dist = 7
features = compute_features(p1, dist)

size = features.shape[0]
reduced_features = features[:, 64:]
h1 = h_array[3, :N]
factor = np.max(np.abs(h1[64:]))
h0 = h1[64:] / factor
t0 = 0.0005 * np.random.randn(2 * size + 2)
sol, nit, rc = fmin_tnc(lambda t: cost_function4(t, h0, reduced_features), t0)

nl = (sol[:size] + 1j * sol[size:2 * size]) * factor
d = (sol[2 * size] + 1j * sol[2 * size + 1]) * factor

est = nl @ reduced_features + d
error = np.sum(np.abs(h0 * factor - est) ** 2) / np.sum(np.abs(h0 * factor - np.average(h0 * factor)) ** 2)
print(f"Error considering nonlinearities: {error}")
print(f"Noise power: {1 / 2 * np.average(np.abs(est - h0 * factor) ** 2) / (141 ** 2)}")

plt.plot(np.abs(est))
plt.plot(np.abs(h0 * factor))
