from scipy.io import loadmat
import numpy as np

from solution.model1.funcs import rate, K, M, B

# Get non-linear and linear solutions
theta_nl = loadmat('../model_solution/solution1_non_linear.mat')['theta']
theta_l = loadmat('../model_solution/solution2_linear.mat')['theta']
leaderboard = np.genfromtxt('../../../datasets/Leaderboard1.csv',
                            delimiter=';',
                            skip_header=True)[:, 2:]

# Get UnBounded's real rates and real score
real_rates = leaderboard[1, 1:]
real_total_rate = leaderboard[1, 0]
competition = np.delete(leaderboard, 1, 0)[:, 1:]

# Get the estimated rates and the estimation error
error = np.zeros((9, 50))
rates = np.zeros((9, 50))
for complexity in range(9):
    for u in range(50):
        rates[complexity, u] = 1e-6 * B / (K + M - 1) * rate(u, theta_nl[:, u], complexity)[0]
        error[complexity, u] = (real_rates[u] - rates[complexity, u]) ** 2

# Compute the NLOS users using a threshold
nlos_users = (real_rates < 75).astype(float)

# Compute for each user how many teams have gotten better results
better_than_us_number = np.sum((competition > real_rates).astype(float), axis=0)
better_than_us = better_than_us_number.astype(bool).astype(float)

# Compute for which users our rates were strictly greater than all other teams
better_than_all = np.all((competition < real_rates), axis=0).astype(float)

# Compute for which users out rates were equal to at least one team, without having another team ahead
equal_and_less = (1 - better_than_all) * (1 - better_than_us)

# See if the minimum error is achieved at the linear model
min_error_is_linear = (np.argmin(error, axis=0) == 0).astype(float)

# See for which users the non-linear and linear solutions are equal
linear_solution_is_equal = np.all(theta_l == theta_nl, axis=0).astype(float)

# Check the total rate
total_rate = np.sum((nlos_users + 1) * real_rates) / 50
