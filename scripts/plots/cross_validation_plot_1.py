from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


error_data1 = loadmat("../../datasets/cross_validation_error_1.mat")
error_data2 = loadmat("../../datasets/cross_validation_error_2.mat")
error1 = error_data1['cv_error']
error2 = error_data2['cv_error']

for i in range(20):
    plt.figure()
    plt.plot(error2[i, :])

print(np.argmin(error1, axis=1))
print(np.argmin(error2, axis=1))
