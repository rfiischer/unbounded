import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from scipy.io import loadmat

from solution.model2.funcs import rate, upper_bound, snr, K, M, B

# Get leaderboard
theta = loadmat('../model2/model_solution/solution_gen_6.mat')['theta'].reshape(50, 4096).T
leaderboard = np.genfromtxt('../../datasets/Leaderboard2.csv',
                            delimiter=';',
                            skip_header=True)[:, 2:]
real_rates = leaderboard[1, 1:]
nlos = (real_rates < 75).astype(float)
nlos_users = np.where(nlos)[0]
los_users = np.where(nlos == 0)[0]

# Compute upper bound and snr
ub = np.zeros(50)
snr_vec = np.zeros(50)
rate_vec = np.zeros(50)

for u in range(50):
    rate_vec[u] = 1e-6 * B / (K + M - 1) * rate(u, theta[:, u])[0]
    ub[u] = 1e-6 * B / (K + M - 1) * upper_bound(u)
    snr_vec[u] = snr(u, theta[:, u])

# Plot
fig1, (ax1, ax2) = plt.subplots(2, 1)
ax1.stem(nlos_users + 1, rate_vec[nlos_users] / ub[nlos_users],
         'C1', markerfmt='C1o', bottom=0.6, label="NLOS Users", basefmt="k")
ax1.stem(los_users + 1, rate_vec[los_users] / ub[los_users],
         'C0', markerfmt='C0o', bottom=0.6, label="LOS Users", basefmt="k")

ax1.set_ylim([0.55, 1.05])
ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax1.set_ylabel("% of Upper Bound Achieved")
ax1.grid()
ax1.legend(ncol=2)

ax2.stem(nlos_users + 1, snr_vec[nlos_users],
         'C1', markerfmt='C1o', bottom=10, label="NLOS Users", basefmt="k")
ax2.stem(los_users + 1, snr_vec[los_users],
         'C0', markerfmt='C0o', bottom=10, label="LOS Users", basefmt="k")
ax2.set_xticks([1, 10, 20, 30, 40, 50])
ax2.set_xlabel("User ID")
ax2.set_ylabel("SNR (dB)")
ax2.grid()

fig2, ax = plt.subplots()
ax.stem(np.arange(1, 51), ub,
        'C1', markerfmt='C1o', bottom=0, label="Upper Bound", basefmt="k")

ax.stem(np.arange(1, 51), rate_vec,
        'C0', markerfmt='C0o', bottom=0, label="Achieved (estimated)", basefmt="k")
ax.set_xticks([1, 10, 20, 30, 40, 50])
ax.set_xlabel("User ID")
ax.set_ylabel("Rate (Mbps)")
ax.grid()
ax.set_ylim([-5, 165])
ax.legend(ncol=2)
