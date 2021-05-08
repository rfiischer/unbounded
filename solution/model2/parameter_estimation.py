import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import fmin_tnc

from functions import compute_features_1

# Load data
data = loadmat("../../datasets/h_user.mat")

# Load variables
s = 64
h_user = data['h_user']
M = data['M'][0, 0]
N = data['N'][0, 0]
K = data['K'][0, 0]
pilotMatrix = np.float64(data['pilotMatrix4N'])

# Compute features
features = compute_features_1(pilotMatrix, 0)[1:, :64]
inv = np.linalg.inv(features)
fsize = features.shape[0] + 1

# Train model on each user
print("----TRAINING----")
model = np.zeros((50, 20, fsize), dtype=complex)
for u in range(50):
    for k in range(20):
        # Log
        print(u, k)

        # Find solution
        h1 = h_user[u, k, :]
        factor = np.max(np.abs(h1))
        h0 = h1 / factor

        # Get direct path
        d = np.average(h0)

        # Get nonlinear behaviour
        na = h0.reshape((4, -1))
        nb = np.zeros(N // 4, dtype=complex)
        n = np.average(na[1:, :], axis=0)[:64] - d
        hl = h0[:64] - n - d

        # Get solution
        c = inv @ hl * factor
        d = d * factor

        model[u, k, :fsize] = np.concatenate(([d], c))

savemat('model_solution/user_model2.mat', {'M': M, 'N': N, 'K': K, 's': s,
                                           'pilotMatrix': pilotMatrix,
                                           'user_model': model,
                                           'fsize': fsize})
