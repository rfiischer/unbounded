from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat("model_solution/user_model2.mat")

model = data['user_model']

u = 19
tap = 1

fig, ax = plt.subplots()
c = model[u, tap, :]
ax.scatter(c.real, c.imag)

ax.set_aspect('equal')
