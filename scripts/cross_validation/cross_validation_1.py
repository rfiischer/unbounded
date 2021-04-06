import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import fmin_tnc

from functions import cost_function4, compute_features

# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']
pilotMatrix4N = np.float64(pilotMatrix4N)
p1 = pilotMatrix4N[:, :N]

error = np.zeros((20, 9))
noise = np.zeros((20, 9))
for dist in range(0, 9):

    # Get nonlinear features
    features = compute_features(p1, dist)
    test_features = compute_features(pilotMatrix4N, dist)

    for k in range(20):
        # Log
        print(k, dist)

        # Find solution
        size = features.shape[0]
        h1 = h_array[k, :N]
        factor = np.max(np.abs(h1))
        h0 = h1 / factor
        t0 = 0.0005 * np.random.randn(2 * size + 2)
        sol, nit, rc = fmin_tnc(lambda t: cost_function4(t, h0, features), t0)

        nl = (sol[:size] + 1j * sol[size:2 * size]) * factor
        d = (sol[2 * size] + 1j * sol[2 * size + 1]) * factor

        est = nl @ test_features + d
        error[k, dist] = np.sum(np.abs(h_array[k, N:] - est[N:]) ** 2) / \
                         np.sum(np.abs(h_array[k, N:] - np.average(h_array[k, N:])) ** 2)

        noise[k, dist] = 1 / 2 * np.average(np.abs(est[N:] - h_array[k, N:]) ** 2) / 20000

savemat("../../datasets/cross_validation_error_1.mat", {'cv_error': error, 'cv_noise': noise})
