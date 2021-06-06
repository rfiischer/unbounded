from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from solution.model2.funcs import snr

data = loadmat("model_solution/user_model2.mat")
theta = loadmat('../model2/model_solution/solution_gen_6.mat')['theta'].reshape(50, 4096).T

leaderboard = np.genfromtxt('../../datasets/Leaderboard2.csv',
                            delimiter=';',
                            skip_header=True)[:, 2:]
real_rates = leaderboard[1, 1:]
nlos = (real_rates < 75).astype(float)
nlos_users = np.where(nlos)[0]
los_users = np.where(nlos == 0)[0]

model = data['user_model']

tap = 1

fig, ax = plt.subplots(5, 10)
for i in range(5):
    for j in range(10):
        c = model[10 * i + j, tap, :]
        if 10 * i + j in nlos_users:
            color = "C1"
            marker = "o"

        else:
            color = "C0"
            marker = "^"

        signal_to_noise = snr(10 * i + j, theta[:, 10 * i + j])

        ax[i, j].scatter(c[1:].real, c[1:].imag, c=color, marker=marker)
        lim = np.max([np.max(np.abs(ax[i, j].get_ylim())), np.max(np.abs(ax[i, j].get_xlim()))])
        ax[i, j].set_xlim([-lim, lim])
        ax[i, j].set_ylim([-lim, lim])
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_title(f"User ID {10 * i + j + 1} SNR {int(signal_to_noise)}", size=10)
        ax[i, j].set_aspect('equal')

plt.subplots_adjust(left=0.02, bottom=0.03, right=0.98, top=0.955, wspace=0.2, hspace=0.264)
