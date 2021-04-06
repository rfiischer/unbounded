from scipy.io import loadmat
from scipy.io import savemat
from scipy.linalg import dft
import numpy as np


# Load dataset
dataset2 = loadmat("../../datasets/dataset2.mat")


# Unpack constants
K = dataset2['K'][0, 0].astype(int)
M = dataset2['M'][0, 0].astype(int)
N = dataset2['N'][0, 0].astype(int)


# Unpack matrices
pilotMatrix = dataset2['pilotMatrix']
receivedSignal = np.transpose(dataset2['receivedSignal'], axes=[2, 0, 1])
transmitSignal = dataset2['transmitSignal']
del dataset2


# Estimate channel
F = dft(K)[:, :M]
iF = dft(K).conj()[:M, :] / K
h_array = iF @ receivedSignal / transmitSignal[0, 0]
hf_array = F @ h_array

savemat('../../datasets/h_user.mat', {'h_user': h_array, 'M': M, 'K': K, 'N': N, 'pilotMatrix4N': pilotMatrix})
del h_array

# Estimate variance
print("Estimating variance.")
var = 1 / 2 * np.average(np.abs(receivedSignal - transmitSignal * hf_array) ** 2, axis=1)
savemat('../../datasets/var_user.mat', {'var_user': var, 'M': M, 'K': K, 'N': N, 'pilotMatrix4N': pilotMatrix})
del receivedSignal

# Estimate SNR
print("Estimating SNR.")
snr = 10 * np.log10(np.average(np.abs(transmitSignal * hf_array) ** 2, axis=1) / (2 * var))
savemat('../../datasets/snr_user.mat', {'snr_user': snr, 'M': M, 'K': K, 'N': N, 'pilotMatrix4N': pilotMatrix})
del snr
