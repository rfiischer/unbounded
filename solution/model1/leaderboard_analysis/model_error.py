from scipy.io import loadmat
import numpy as np

from solution.model1.funcs import rate, K, M, B

theta = loadmat('../model_solution/solution1_non_linear.mat')['theta']
leaderboard = np.genfromtxt('../../../datasets/Leaderboard1.csv',
                            delimiter=';',
                            skip_header=True)[:, 2:]

real_rates = leaderboard[1, 1:] * 1e6
real_total_rate = leaderboard[1, 0] * 1e6

error = np.zeros((9, 50))
rates = np.zeros((9, 50))
for complexity in range(9):
    for u in range(50):
        rates[complexity, u] = B / (K + M - 1) * rate(u, theta[:, u], complexity)[0]
        error[complexity, u] = (real_rates[u] - rates[complexity, u]) ** 2
