from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat("model_solution/user_model1.mat")

model = data['user_model']
lengths = data['lengths'][0].astype(int)
lengths = np.concatenate(([0], lengths))

u = 19
tap = 1

fig, ax = plt.subplots()
for i in range(len(lengths) - 1):
    ci = model[u, tap, lengths[i]:lengths[i + 1]]
    ax.scatter(ci.real, ci.imag)

ax.set_aspect('equal')
