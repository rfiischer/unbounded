from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


# Load dataset
snr_data = loadmat("..\\datasets\\snr_user.mat")
h_data = loadmat("..\\datasets\\h_user.mat")


# Unpack constants
K = h_data['K'][0, 0].astype(int)
M = h_data['M'][0, 0].astype(int)
N = h_data['N'][0, 0].astype(int)


# Get values
snr = snr_data['snr_user']
h_user = h_data['h_user']


# Plot
k = 3
fig, ax = plt.subplots(5, 10, sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax[i, j].plot(np.abs(h_user[10 * i + j, k, :]))
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].set_ylim(-0.1e-5, 1e-5 + 0.1e-5)

plt.subplots_adjust(0.079, 0.11, 0.95, 0.88, 0.455, 0.2)

fig, ax = plt.subplots(5, 10, sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax[i, j].plot(np.abs(snr[10 * i + j, :]))
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].set_ylim(0.7, 41)

plt.subplots_adjust(0.079, 0.11, 0.95, 0.88, 0.455, 0.2)
