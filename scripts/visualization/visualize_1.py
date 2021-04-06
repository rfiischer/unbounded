import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Load data
data = loadmat("../../datasets/h_estimated.mat")

# Load variables
h_array = data['h_array']
M = data['M'][0, 0]
N = data['N'][0, 0]
pilotMatrix4N = data['pilotMatrix4N']

# # Choose N random configurations
# perm = np.random.permutation(4 * N)[:N]
# h_sel = h_array[:, perm]
# configs = pilotMatrix4N[:, perm]

# Choosing N random configurations unfortunately almost allways results in a singular matrix
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

plt.figure()
plt.plot(h1.real)
plt.plot(h2.real)

plt.figure()
plt.plot(h1.imag)
plt.plot(h2.imag)

plt.figure()
plt.plot(h1.real)
plt.plot(h3.real)

plt.figure()
plt.plot(h1.real)
plt.plot(h4.real)

plt.figure()
plt.plot(np.abs(h1))
plt.plot(np.abs(h2))

plt.figure()
plt.plot(np.abs(h1))
plt.plot(np.abs(h3))

plt.figure()
plt.plot(np.abs(h1))
plt.plot(np.abs(h4))

plt.figure()
plt.plot(np.angle(h1))
plt.plot(np.angle(h2))

plt.figure()
plt.plot(np.angle(h1))
plt.plot(np.angle(h3))

plt.figure()
plt.plot(np.angle(h1))
plt.plot(np.angle(h4))
