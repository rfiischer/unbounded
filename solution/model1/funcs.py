from scipy.io import loadmat
from scipy.fft import fft
import numpy as np
import os

from functions import compute_features_1


# Make relative paths work by loading script from everywhere
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "user_model1.mat")
data = loadmat(filename)

# Load parameters
P = 1
B = 10e6
N0 = 3.1613e-20
K = data['K'][0, 0]
M = data['M'][0, 0]
s = data['s'][0, 0]
model = data['user_model']
lengths = data['lengths'][0].astype(int)
idxs = data['idxs'][0].astype(int)

# Retrieve feature indexes
sep_idxs = []
for dist in range(len(lengths)):
    start = sum(lengths[:dist])
    sep_idxs.append(np.array(idxs[start:start + lengths[dist]]))


# Define functions
def rate(user, theta, complexity):

    size = lengths[complexity + 1]
    features = compute_features_1(theta, complexity)

    ht = model[user, :, :size] @ features[sep_idxs[complexity + 1]]
    hf = fft(ht, n=K, axis=0)

    # Return normalized rates
    r = np.sum(np.log2(1 + P * np.abs(hf) ** 2 / (B * N0)), axis=0)

    return r, ht, hf


def optimize1(user, n_angles=8, max_complexity=0):
    # Get linear coefficients
    d = model[user, :, 0].reshape(-1, 1)
    c = model[user, :, 1:65]
    central_direction = np.angle(d)

    # For each direction and channel tap compute rate
    rates = np.zeros((M, n_angles))
    for angle in range(n_angles):
        solutions = np.tile(np.sign((c * np.conj(np.exp(1j * 2 * np.pi / n_angles * angle +
                                                        1j * central_direction))).real), (1, 64))

        for i in range(M):
            rates[i, angle], ht, hf = rate(user, solutions[i, :], max_complexity)

    # Find maximum
    max_row, max_col = np.where(rates == np.max(rates))

    return np.tile(np.sign((c[max_row[0], :] * np.conj(np.exp(1j * 2 * np.pi / n_angles * max_col[0] +
                                                              1j * central_direction[max_row[0]]))).real), 64)
