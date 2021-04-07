import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt

from functions import cost_function5, compute_features


# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']


# Get nonlinear features
pilotMatrix4N = np.float64(pilotMatrix4N)
dist = 2
test_features = compute_features(pilotMatrix4N, dist)

# Training size
ts = 3 * N // 4
features = test_features[:, :ts]

k = 3
size = features.shape[0]
h1 = h_array[k, :ts]
factor = np.max(np.abs(h1))
h0 = h1 / factor
t0 = 0.0005 * np.random.randn(2 * size)
sol, nit, rc = fmin_tnc(lambda t: cost_function5(t, h0, features), t0)

c = (sol[:size] + 1j * sol[size:]) * factor

est = c @ test_features
error = np.sum(np.abs(h_array[k, N:] - est[N:]) ** 2) / np.sum(np.abs(h_array[k, N:] - np.average(h_array[k, N:])) ** 2)
print(f"Error considering nonlinearities: {error}")
print(f"Noise power: {1 / 2 * np.average(np.abs(est[N:] - h_array[k, N:]) ** 2) / (141 ** 2)}")

plt.plot(np.abs(est))
plt.plot(np.abs(h_array[k, :]))
