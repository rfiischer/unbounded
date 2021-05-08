import numpy as np
from scipy.io import savemat

from solution.model2.funcs import optimize1

n_angles = 1024

solution_linear = np.zeros((4096, 50))
for i in range(50):
    print(f"Linear, user {i}")
    solution_linear[:, i] = optimize1(i, n_angles)

savemat('model_solution/solution3_linear.mat', {'theta': solution_linear})
