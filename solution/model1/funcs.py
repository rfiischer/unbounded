from scipy.io import loadmat
from scipy.fft import fft
import numpy as np
import os
import matplotlib.pyplot as plt

from functions import compute_features_1, features_sizes_1


# Make relative paths work by loading script from everywhere
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "user_model1.mat")
data = loadmat(filename)

P = 1
B = 10e6
N0 = 3.1613e-20
K = data['K'][0, 0]
M = data['M'][0, 0]
s = data['s'][0, 0]
model = data['user_model']


def rate(user, theta, complexity):

    size = features_sizes_1(s, complexity)
    features = compute_features_1(theta, complexity).flatten()

    ht = model[user, :, :size] @ features
    hf = fft(ht, n=K)

    r = B / (K + M - 1) * np.sum(np.log2(1 + P * np.abs(hf) ** 2 / (B * N0)))

    return r, ht, hf


def optimize1(user):
    d = model[user, :, 0].reshape(-1, 1)
    c = model[user, :, 1:65]

    solutions = np.tile(np.sign((c * np.conj(d)).real), (1, 64))

    rates = np.zeros(M)
    for i in range(M):
        fig, ax = plt.subplots(2, 1)
        rates[i], ht, hf = rate(user, solutions[i, :], 4)
        ax[0].plot(np.abs(ht))
        ax[1].plot(np.abs(hf))

    return solutions[np.argmax(rates), :]


optimize1(19)
