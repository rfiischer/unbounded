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

k = 3
h1 = h_array[k, :N]
h2 = h_array[k, N:2 * N]
h3 = h_array[k, 2 * N: 3 * N]
h4 = h_array[k, 3 * N:]


# Plot
plt.figure()
plt.plot(np.abs((h1 + h2) / 2))
plt.plot(np.abs((h3 + h4) / 2))

plt.figure()
plt.plot(np.angle((h1 + h2) / 2))
plt.plot(np.angle((h3 + h4) / 2))

plt.figure()
plt.plot(np.abs((h1 - h2) / 2))
plt.plot(np.abs((h3 - h4) / 2))

plt.figure()
plt.plot(np.mod(np.angle((h1 - h2) / 2), np.pi))
plt.plot(np.mod(np.angle((h3 - h4) / 2), np.pi))
