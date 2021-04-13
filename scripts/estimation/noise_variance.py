from scipy.io import loadmat
import numpy as np


# Load dataset
dataset1 = loadmat("../../datasets/dataset1.mat")


# Unpack constants
K = dataset1['K'][0, 0].astype(int)
M = dataset1['M'][0, 0].astype(int)
N = dataset1['N'][0, 0].astype(int)


# Unpack matrices
pilotMatrix4N = dataset1['pilotMatrix4N']
receivedSignal4N = dataset1['receivedSignal4N']
transmitSignal = dataset1['transmitSignal']


# Estimate noise using the fact that the same configuration was transmitted twice
# This results in an unbiased estimator
same_config_index = np.zeros(2 * N, dtype=int)
for i in range(2 * N):
    if np.all(pilotMatrix4N[:, i] == pilotMatrix4N[:, i % N + 2 * N]):
        same_config_index[i] = i % N + 2 * N

    else:
        same_config_index[i] = i % N + 3 * N

var = np.zeros(2 * N)
for i in range(2 * N):
    config_a = receivedSignal4N[:, i]
    config_b = receivedSignal4N[:, same_config_index[i]]

    noise = (config_b - config_a) / np.sqrt(2)

    var[i] = np.average(np.abs(noise) ** 2)

print(f"Estimated N0: {np.average(var)}")
