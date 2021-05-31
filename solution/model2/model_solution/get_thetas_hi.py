from scipy.io import loadmat, savemat
import numpy as np
from solution.model2.funcs import rate


def fitness_function_4096(population):
    F = np.zeros(50)
    for user_j in range(50):
        F[user_j] = rate(user_j, population[:, user_j])[0]

    return F


# savemat('solution_gen_5.mat', {'theta': best_config, 'rate': rate_max, 'diff': diff, 'generations': user_generation})
theta_hi = loadmat("solution_gen_5.mat")["theta"]
rate_hi = loadmat("solution_gen_5.mat")["rate"].T
diff_hi = loadmat("solution_gen_5.mat")["diff"].T
generations_hi = loadmat("solution_gen_5.mat")["generations"]

thetas = np.zeros((50, 64, 64))
for i in range(50):
    for j in range(0, 64):
        if theta_hi[i, 0, j] == 1:
            thetas[i, :, j] = 1
        else:
            thetas[i, :, j] = -1

theta_r = np.zeros((4096, 50))
for i in range(50):
    theta_r[:, i] = thetas[i, :, :].reshape(4096)

check = fitness_function_4096(theta_r)

savemat("solution4_genetic.mat", {"theta": theta_r})
