import matplotlib.pyplot as plt
import numpy as np

from solution.model1.funcs import optimize1, rate, model
from functions import complex_plot

# Compute solution
u = 19
n_angle = 64
max_complexity = 0
opt = optimize1(u, n_angle, max_complexity)
r, ht, hf = rate(u, opt, max_complexity)

# Display
print(f"Rate: {r}")

plt.figure()
plt.plot(np.abs(ht))

plt.figure()
plt.plot(np.abs(hf))

plt.figure()
plt.imshow(opt.reshape(64, 64))

plt.figure()
complex_plot([model[u, 0, 0]])
complex_plot(model[u, 0, 1:65] * opt[:64])
