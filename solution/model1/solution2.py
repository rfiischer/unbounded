import numpy as np
from scipy.io import savemat

from solution.model1.funcs import optimize1

n_angles = 256

solution_linear = np.zeros((4096, 50))
for i in range(50):
    print(f"Linear, user {i}")
    solution_linear[:, i] = optimize1(i, n_angles, 0)

savemat('model_solution/solution2_linear.mat', {'theta': solution_linear})
