from scipy.io import loadmat
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

import os

# Make relative paths work by loading script from everywhere
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "..\\datasets\\dataset1.mat")


# Load dataset
dataset1 = loadmat(filename)


# Unpack constants
K = dataset1['K'][0, 0].astype(int)
M = dataset1['M'][0, 0].astype(int)
N = dataset1['N'][0, 0].astype(int)


# Unpack matrices
pilotMatrix4N = dataset1['pilotMatrix4N']
receivedSignal4N = dataset1['receivedSignal4N']
transmitSignal = dataset1['transmitSignal']


# Get pilot symbols in time
# The IFFT from numpy multiplies by 1/K, hence we multiply by sqrt(K) to get 1/sqrt(K)
x = np.sqrt(K) * fft.ifft(transmitSignal[:, 0])


# Cyclic prefix
cp = x[-M:]


# Append cyclic prefix
x_cp = np.concatenate((cp, x))


# Build pilot correlation matrix
R = np.zeros((M, M), dtype=complex)
for i in range(M):
    for j in range(M):
        R[i, j] = np.correlate(x_cp[M - 1 - j:M - 1 - j + K], x_cp[M - 1 - i:M - 1 - i + K])

R_inv = np.linalg.inv(R)

# Loop through all training examples
h_array = np.zeros((20, 4 * N), dtype=complex)
for theta in range(4 * N):

    # Select one of the outputs
    zf = receivedSignal4N[:, theta]
    z = np.sqrt(K) * fft.ifft(zf)

    # Create correlation vector
    r = np.zeros(M, dtype=complex)
    for i in range(M):
        r[i] = np.correlate(z, x_cp[M - 1 - i:M - 1 - i + K])

    # Estimate h
    h = np.matmul(R_inv, r)

    # Store estimate
    h_array[:, theta] = h


# Plot the example h with maximum peak value
ex = np.argmax(np.max(np.abs(h_array), axis=0))
h = h_array[:, ex]
fig, [ax1, ax2] = plt.subplots(2, 1)
ax1.stem(np.abs(h))
ax1.set_ylabel('Amplitude')
ax1.set_xticks(np.arange(0, M))

ax2.stem(np.angle(h) / np.pi * 180)
ax2.set_ylabel('Phase')
ax2.set_xticks(np.arange(0, M))

ax1.set_title(f'Configuration with peak h: {ex}')

plt.show()
