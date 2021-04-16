import numpy as np
from scipy.io import savemat

from solution.model1.funcs import optimize1, rate

n_angles = 64

solution_linear = np.zeros((4096, 50))
for i in range(50):
    print(f"Linear, user {i}")
    solution_linear[:, i] = optimize1(i, n_angles, 0)

solution_non_linear = np.zeros((4096, 50))
for i in range(50):
    print(f"Non-linear, user {i}")
    solution_non_linear[:, i] = optimize1(i, n_angles, 8)

savemat('solution1_linear.mat', {'theta': solution_linear})
savemat('solution1_non_linear.mat', {'theta': solution_non_linear})

# Compare the effect of non-linear features in linear optimization and vice versa
rates_linear_linear = np.zeros(50)
for u in range(50):
    rates_linear_linear[u] = rate(u, solution_linear[:, u], 0)[0]

rates_non_linear_non_linear = np.zeros(50)
for u in range(50):
    rates_non_linear_non_linear[u] = rate(u, solution_non_linear[:, u], 8)[0]

rates_linear_non_linear = np.zeros(50)
for u in range(50):
    rates_linear_non_linear[u] = rate(u, solution_linear[:, u], 8)[0]

rates_non_linear_linear = np.zeros(50)
for u in range(50):
    rates_non_linear_linear[u] = rate(u, solution_non_linear[:, u], 0)[0]
