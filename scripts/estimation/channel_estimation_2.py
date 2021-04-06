from scipy.io import loadmat
from scipy.io import savemat
from scipy.linalg import dft
import numpy as np
import matplotlib.pyplot as plt


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


# Estimate channel
F = dft(K)[:, :M]
iF = dft(K).conj()[:M, :] / K
h_array = iF @ receivedSignal4N / transmitSignal[0, 0]
hf_array = F @ h_array


# Estimate variance
z_est = transmitSignal * hf_array
var = 1 / 2 * np.average(np.abs(receivedSignal4N - z_est) ** 2, axis=0)
power = np.average(np.abs(z_est) ** 2, axis=0)
snr = 10 * np.log10(power / (2 * var))


savemat('../../datasets/h_estimated.mat', {'h_array': h_array, 'hf_array': hf_array,
                                           'M': M, 'K': K, 'N': N, 'pilotMatrix4N': pilotMatrix4N})

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

# Plot SNR
fig, ax = plt.subplots()
ax.plot(snr)
ax.set_ylabel('SNR (dB)')
