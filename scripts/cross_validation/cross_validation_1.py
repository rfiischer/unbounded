import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import fmin_tnc

from functions import cost_function5, compute_features, features_sizes

# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
s = 64
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']
pilotMatrix4N = np.float64(pilotMatrix4N)
p1 = pilotMatrix4N[:, :N]

error = np.zeros((20, 9))
noise = np.zeros((20, 9))
test_features = compute_features(pilotMatrix4N, 8)
features = test_features[:, :N]
for dist in range(0, 9):

    # Get nonlinear features
    fsize = features_sizes(s, dist)
    features_d = features[:fsize, :]
    test_features_d = test_features[:fsize, :]

    for k in range(20):
        # Log
        print(k, dist)

        # Find solution
        size = features_d.shape[0]
        h1 = h_array[k, :N]
        factor = np.max(np.abs(h1))
        h0 = h1 / factor
        t0 = 0.0005 * np.random.randn(2 * size)
        sol, nit, rc = fmin_tnc(lambda t: cost_function5(t, h0, features_d), t0)

        c = (sol[:size] + 1j * sol[size:]) * factor

        est = c @ test_features_d
        error[k, dist] = np.sum(np.abs(h_array[k, N:] - est[N:]) ** 2) / \
                         np.sum(np.abs(h_array[k, N:] - np.average(h_array[k, N:])) ** 2)

        noise[k, dist] = 1 / 2 * np.average(np.abs(est[N:] - h_array[k, N:]) ** 2) / 20000

savemat("../../datasets/cross_validation_error_1.mat", {'cv_error': error, 'cv_noise': noise})
