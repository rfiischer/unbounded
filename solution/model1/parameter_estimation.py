import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import fmin_tnc

from functions import cost_function5, compute_features_1, features_sizes_1, li_features

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
ts = 3 * N // 4  # Training set size
mo = 8  # Max model order
features = compute_features_1(pilotMatrix, mo)
test_features = features[:, ts:]
train_features = features[:, :ts]

# Perform cross-validation on user data
print("----CROSS-VALIDATION----")
error = np.zeros((50, 20, mo + 2))
lengths = np.zeros(mo + 2)
idxs = []
for u in range(50):
    for dist in range(-1, mo + 1):

        # Get nonlinear features
        complete_size = features_sizes_1(s, dist)
        li_idx = li_features(features[:complete_size, :])
        idxs.extend(li_idx)
        train_features_d = train_features[li_idx, :]
        test_features_d = test_features[li_idx, :]
        fsize = len(li_idx)
        lengths[dist + 1] = fsize

        for k in range(20):
            # Log
            print(u, dist, k)

            # Find solution
            h1 = h_user[u, k, :ts]
            factor = np.max(np.abs(h1))
            h0 = h1 / factor
            t0 = 0.0005 * np.random.randn(2 * fsize)
            sol, nit, rc = fmin_tnc(lambda t: cost_function5(t, h0, train_features_d), t0, disp=0)

            c = (sol[:fsize] + 1j * sol[fsize:]) * factor

            h_test_est = c @ test_features_d
            h_test = h_user[u, k, ts:]
            error[u, k, dist + 1] = np.sum(np.abs(h_test - h_test_est) ** 2) / \
                                    np.sum(np.abs(h_test - np.average(h_test)) ** 2)

# Find ideal complexity
user_max = np.max(np.abs(h_user), axis=-1)
best_complexity = np.where(user_max > 1.5e-7, np.argmin(error, axis=-1), 0)

# Train model on each user
print("----TRAINING----")
li_idx = li_features(features)
model = np.zeros((50, 20, len(li_idx)), dtype=complex)
for u in range(50):
    for k in range(20):
        # Log
        print(u, k)

        # Get nonlinear features
        complexity = best_complexity[u, k] - 1
        complete_size = features_sizes_1(s, complexity)
        li_idx = li_features(features[:complete_size, :])
        features_d = features[li_idx, :]
        fsize = len(li_idx)

        # Find solution
        h1 = h_user[u, k, :]
        factor = np.max(np.abs(h1))
        h0 = h1 / factor
        t0 = 0.0005 * np.random.randn(2 * fsize)
        sol, nit, rc = fmin_tnc(lambda t: cost_function5(t, h0, features_d), t0, disp=0)

        model[u, k, :fsize] = (sol[:fsize] + 1j * sol[fsize:]) * factor

savemat('model_solution/user_model1.mat', {'M': M, 'N': N, 'K': K, 's': s, 'ts': ts,
                                           'pilotMatrix': pilotMatrix,
                                           'user_model': model,
                                           'best_complexity': best_complexity,
                                           'error': error,
                                           'lengths': lengths,
                                           'idxs': idxs})
