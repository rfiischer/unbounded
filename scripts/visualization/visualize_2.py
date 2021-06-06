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

k = 2
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

fig, ax = plt.subplots()
ax.plot((h1 + h2) / 2)
ax.set_ylabel(f'$|h_1^{k} + h_2^{k}| / 2$')
ax.set_xlabel('Configuration Index $v$', fontsize=12)

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.abs((h1 + h2) / 2), label=f'$|h_1^{k} + h_2^{k}| / 2$')
ax[0].plot(np.abs((h3 + h4) / 2), label=f'$|h_3^{k} + h_4^{k}| / 2$')

ax[1].plot(np.abs((h1 - h2) / 2), label=f'$|h_1^{k} - h_2^{k}| / 2$')
ax[1].plot(np.abs((h3 - h4) / 2), label=f'$|h_3^{k} - h_4^{k}| / 2$')
ax[1].set_xlabel('Configuration Index $v$', fontsize=12)
ax[1].set_ylabel('b) \n Odd Order', fontsize=12)
ax[0].set_ylabel('a) \n Even Order', fontsize=12)
ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
ax[0].set_title(fr'Even Order/Odd Order Terms of $h_\theta^v[{k}]$')
ax[0].set_xticks([])

fig, ax = plt.subplots()
hn = np.abs((h1 + h2) / 2)
ax.plot(hn[:63], label=r'$|h_1^1 + h_2^1|[0:64] \,/\, 2$')
ax.plot(hn[1024:1087], label=r'$|h_1^1 + h_2^1|[1024:1088] \,/\, 2$')
ax.plot(hn[2048:2111], label=r'$|h_1^1 + h_2^1|[2048:2112] \,/\, 2$')
ax.plot(hn[3072:3135], label=r'$|h_1^1 + h_2^1|[3072:3136] \,/\, 2$')
ax.legend(ncol=2, fontsize=10, loc='upper left')
ax.set_title(r'Repetition of Even-Order Terms')
ax.set_xlabel(r'Configuration Index $v$ (mod $1024$)', fontsize=12)
ax.set_ylabel(r'$e(\theta^v, 1)$', fontsize=12)

fig, ax = plt.subplots(2, 2, figsize=(5, 5))
for i in range(2):
    for j in range(2):
        ax[i, j].imshow(p1[np.random.randint(0, 64)].reshape(64, 64))
        ax[i, j].tick_params(axis="x", labelsize=12)
        ax[i, j].tick_params(axis="y", labelsize=12)

plt.subplots_adjust(left=0.07, bottom=0.064, right=0.97, top=0.964)

fig, ax = plt.subplots(2, 2, figsize=(5, 5))
label = ['f', 'g', 'h']
for i in range(2):
    for j in range(2):
        if (2 * i + j) != 3:
            ax[i, j].imshow((p1[:, 0] * p1[:, (2 * i + j + 1) * 1024]).reshape(64, 64))
            ax[i, j].set_ylabel(label[(2 * i + j)], fontdict={'size': 16})
            ax[i, j].tick_params(axis="x", labelsize=12)
            ax[i, j].tick_params(axis="y", labelsize=12)

plt.subplots_adjust(left=0.07, bottom=0.064, right=0.97, top=0.964)
ax[1, 1].set_visible(False)
