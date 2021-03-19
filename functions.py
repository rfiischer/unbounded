import matplotlib.pyplot as plt
import numpy as np


def complex_plot(a, style='ro-'):

    for ai in range(len(a)):
        plt.plot([0, a[ai].real], [0, a[ai].imag], style)

    plt.ylabel('Imaginary')
    plt.xlabel('Real')


def cost_function1(t, h, m1, m2):

    # Get params
    n, d = m1.shape

    # Get scale factor (when theta = 1)
    a1 = t[0]

    # Get real and imaginary parts of Cn
    re = t[1:n + 1]
    im = t[n + 1:]

    # Scale theta when theta == 1
    x = a1 * m1 - m2

    # Compute real and imaginary parts of h
    reh = x.T @ re
    imh = x.T @ im

    # Compute usefull vectors
    r_e = (reh - h.real)
    i_e = (imh - h.imag)

    # Compute cost
    cost = 1 / 2 * (np.sum(r_e ** 2) + np.sum(i_e ** 2))

    # Compute gradient
    gradient = np.zeros_like(t)
    gradient[0] = r_e @ (m1.T @ re) + i_e @ (m1.T @ im)
    gradient[1:n + 1] = r_e @ x.T
    gradient[n + 1:] = i_e @ x.T

    return cost, gradient
