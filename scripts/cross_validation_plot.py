from scipy.io import loadmat
import matplotlib.pyplot as plt


error_data = loadmat("..\\datasets\\cross_validation_error.mat")
error = error_data['cv_error']

for i in range(20):
    plt.figure()
    plt.plot(error[i, :])
