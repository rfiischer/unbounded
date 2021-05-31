from scipy.io import loadmat
from scipy.fft import fft
from scipy.linalg import dft
import numpy as np
import os

from functions import compute_features_1


# Make relative paths work by loading script from everywhere
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "model_solution/user_model2.mat")
data = loadmat(filename)

# Load parameters
P = 1
B = 10e6
N0 = 3.1613e-20
K = data['K'][0, 0]
M = data['M'][0, 0]
s = data['s'][0, 0]
F = dft(K)[:, :M]
model = data['user_model']
fsize = data['fsize']


# Define functions
def rate(user, theta):

    features = compute_features_1(theta, 0)

    ht = model[user, :, :] @ features
    hf = fft(ht, n=K, axis=0)

    # Return normalized rates
    r = np.sum(np.log2(1 + P * np.abs(hf) ** 2 / (B * N0)), axis=0)

    return r, ht, hf


def snr(user, theta):

    features = compute_features_1(theta, 0)

    ht = model[user, :, :] @ features
    hf = fft(ht, n=K, axis=0)

    # Return snr
    r = np.average(np.abs(hf) ** 2) / (N0 * B)

    return 10 * np.log10(r)


def upper_bound(user):

    # Convert the channel model to the frequency domain
    scale = np.array([1] + [64] * 64)
    hf = F @ (model[user, :, :] * scale)

    return np.sum(np.log2(1 + P * np.abs(np.sum(np.abs(hf), axis=1)) ** 2 / (B * N0)), axis=0)


def optimize1(user, n_angles=8):
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
            rates[i, angle], ht, hf = rate(user, solutions[i, :])

    # Find maximum
    max_row, max_col = np.where(rates == np.max(rates))

    return np.tile(np.sign((c[max_row[0], :] * np.conj(np.exp(1j * 2 * np.pi / n_angles * max_col[0] +
                                                              1j * central_direction[max_row[0]]))).real), 64)
