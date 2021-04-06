from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


error_data = loadmat("../../datasets/cross_validation_error_1.mat")
error = error_data['cv_error']

for i in range(20):
    plt.figure()
    plt.plot(error[i, :])

print(np.argmin(error, axis=1))
