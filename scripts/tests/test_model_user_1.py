from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from functions import optimize_tap


# Load dataset
h_data = loadmat("../../datasets/h_user.mat")


# Unpack constants
K = h_data['K'][0, 0].astype(int)
M = h_data['M'][0, 0].astype(int)
N = h_data['N'][0, 0].astype(int)
configs = h_data['pilotMatrix4N'].astype(float)


# Get values
h_user = h_data['h_user']


# Plot
k = 2
fig, ax = plt.subplots(5, 10, sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        hu = h_user[10 * i + j, k, :].reshape((4, -1))
        ax[i, j].plot(np.abs(hu[0, :]))
        ax[i, j].plot(np.abs(hu[1, :]))
        ax[i, j].plot(np.abs(hu[2, :]))
        ax[i, j].plot(np.abs(hu[3, :]))
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].set_ylim(-0.1e-5, 1e-5 + 0.1e-5)

plt.subplots_adjust(0.079, 0.11, 0.95, 0.88, 0.455, 0.2)

fig, ax = plt.subplots(5, 10, sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        hu = h_user[10 * i + j, k, :].reshape((4, -1))
        nb = np.zeros(N // 4, dtype=complex)
        nb[:64] = np.average(hu[1:, :], axis=0)[:64]
        nb[64:] = np.average(hu, axis=0)[64:]
        ax[i, j].plot(np.abs(h_user[10 * i + j, k, :] - np.tile(nb, 4)))
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].set_ylim(-0.1e-5, 1e-5 + 0.1e-5)

plt.subplots_adjust(0.079, 0.11, 0.95, 0.88, 0.455, 0.2)

u = 19
hu = h_user[u, k, :]
carr = []
darr = []
sarr = []
optimize_tap(configs, hu, method='nonlinear-average')
methods = ['linear', 'linear-direct', 'nonlinear-simple-single', 'nonlinear-simple-average',
           'nonlinear-single', 'nonlinear-average', 'second-order-simple']
for method in methods:

    c, d, s = optimize_tap(configs, hu, method=method)
    carr.append(c)
    darr.append(d)
    sarr.append(s)

fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(len(methods)):
    ax[i].imshow(sarr[i].real.reshape(64, 64))
    ax[i].set_title(methods[i])

ax[-1].set_visible(False)

fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(len(methods)):
    ax[i].imshow(carr[i].real.reshape(64, 64))
    ax[i].set_title(methods[i])

ax[-1].set_visible(False)
