import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Load data
data = loadmat("..\\datasets\\h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']

# Assemble vectors
pilotMatrix4N = np.float64(pilotMatrix4N)
p1 = pilotMatrix4N[:, :N]
p2 = pilotMatrix4N[:, N:2 * N]
p3 = pilotMatrix4N[:, 2 * N:3 * N]
p4 = pilotMatrix4N[:, 3 * N:]

fig, ax = plt.subplots(2, 4)
for idx, i in enumerate([1200, 2048 + 1200]):
    ax[idx, 0].imshow(p1[:, i].reshape((64, 64)))
    ax[idx, 1].imshow(p2[:, i].reshape((64, 64)))
    ax[idx, 2].imshow(p3[:, i].reshape((64, 64)))
    ax[idx, 3].imshow(p4[:, i].reshape((64, 64)))

idxs = []
cr = pilotMatrix4N.reshape((64, 64, -1))
for i in range(16384):
    if np.all(np.sum(cr[:, :, i], axis=0) != 0):
        idxs.append(i)

true_in_first64 = np.all(idxs == (np.tile(np.arange(64), (4, 1)) + np.array([[0], [N], [2 * N], [3 * N]])).flatten())
print(f"Vertical configurations occur only in first 64 configurations? {true_in_first64}")

fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(p1[:, 4 * i + j].reshape((64, 64)))
