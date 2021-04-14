import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt

from functions import cost_function5, compute_features, features_sizes, li_features


# Load data
data = loadmat("../../datasets/h_user.mat")

# Load variables
u = 19
h_array = data['h_user'][u, :, :]
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix = np.float64(data['pilotMatrix4N'])

# Get nonlinear features
max_dist = 3
complete_features = compute_features(pilotMatrix, max_dist)
dist = 3
complete_size = features_sizes(64, dist)
li_idx = li_features(complete_features[:complete_size, :])
test_features = complete_features[li_idx, :]

# Training size
ts = 3 * N // 4
features = test_features[:, :ts]

k = 2
size = features.shape[0]
h1 = h_array[k, :ts]
factor = np.max(np.abs(h1))
h0 = h1 / factor
t0 = 0.0005 * np.random.randn(2 * size)
sol, nit, rc = fmin_tnc(lambda t: cost_function5(t, h0, features), t0)

c = (sol[:size] + 1j * sol[size:]) * factor

h_est = c @ test_features
error = np.sum(np.abs(h_array[k, ts:] - h_est[ts:]) ** 2) / \
        np.sum(np.abs(h_array[k, ts:] - np.average(h_array[k, ts:])) ** 2)
print(f"Error considering nonlinearities: {error}")

# plt.plot(np.abs(h_array[k, :]))
# plt.plot(np.abs(h_est))

nli = np.concatenate(([0], np.arange(65, 128)))
cnl = c[nli]
fnl = test_features[nli, :]
nlin = cnl @ fnl
