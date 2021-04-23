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

# Assemble vectors
pilotMatrix4N = np.float64(pilotMatrix4N)
p1 = pilotMatrix4N[:, :N]
p2 = pilotMatrix4N[:, N:2 * N]
p3 = pilotMatrix4N[:, 2 * N:3 * N]
p4 = pilotMatrix4N[:, 3 * N:]

k = 1
h1 = h_array[k, :N]
h2 = h_array[k, N:2 * N]
h3 = h_array[k, 2 * N: 3 * N]
h4 = h_array[k, 3 * N:]
lin = (h1 - h2 + p3[0, :] * (h3 - h4)) / 4
nlin = (h1 + h2 + h3 + h4) / 4

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

plt.figure()
plt.plot(np.abs(nlin))

plt.figure()
plt.plot(np.abs(lin))

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.abs((h1 + h2) / 2), label='$|h_1 + h_2| / 2$')
ax[0].plot(np.abs((h3 + h4) / 2), label='$|h_3 + h_4| / 2$')

ax[1].plot(np.abs((h1 - h2) / 2), label='$|h_1 - h_2| / 2$')
ax[1].plot(np.abs((h3 - h4) / 2), label='$|h_3 - h_4| / 2$')
ax[1].set_xlabel('Configuration Index')
ax[1].set_ylabel('$b)$')
ax[0].set_ylabel('$a)$')
ax[0].legend()
ax[1].legend()
ax[0].set_title(r'Even Order/Odd Order Terms of $h_\theta[1]$')
ax[0].set_xticks([])
