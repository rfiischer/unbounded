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
# The IFFT multiplies by 1/K, hence we multiply by sqrt(K) to get 1/sqrt(K)
xf = transmitSignal[:, 0]
x = np.sqrt(K) * fft.ifft(xf)


# Cyclic prefix
cp = x[-(M - 1):]


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
hf_array = np.zeros((K, 4 * N), dtype=complex)
var_array = np.zeros(4 * N)
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
    hf = fft.fft(h, n=K)

    # Store estimate
    h_array[:, theta] = h
    hf_array[:, theta] = hf

    # Find noise variance
    # The noise variance is equal in both frequency and time domain
    # Therefore, it can be estimated in either domain

    # In order to compute the variance, we use our estimation hf * xf for the mean
    # This introduces a bias

    # NOTE: if x is an impulse, we can discard the first samples of z (the ones with channel spread)
    # and compute the noise variance on the remaining samples (which would have zero mean) thus removing the bias

    # Estimate noiseless received signal
    var_array[theta] = 1 / (2 * K) * np.sum(np.abs(zf - hf * xf) ** 2)


# Plot the example h with maximum peak value
ex = np.argmax(np.max(np.abs(h_array), axis=0))
h = h_array[:, ex]
fig1, [ax1, ax2] = plt.subplots(2, 1, sharex='all')
ax1.stem(np.abs(h))
ax1.set_ylabel(r'$|h|$')
ax1.set_xticks(np.arange(0, M))

ax2.stem(np.angle(h) / np.pi * 180)
ax2.set_ylabel(r'$\phi (h)$')
ax2.set_xticks(np.arange(0, M))

hf = hf_array[:, ex]
fig2, [ax3, ax4] = plt.subplots(2, 1, sharex='all')
ax3.plot(np.abs(hf))
ax3.set_ylabel(r'$|H|$')

ax4.plot(np.angle(hf) / np.pi * 180)
ax4.set_ylabel(r'$\phi (H)$')

ax1.set_title(f'Configuration with peak h: {ex}')
ax3.set_title(f'Configuration with peak h: {ex}')

plt.show()
