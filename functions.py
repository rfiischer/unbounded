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


def cost_function2(p, h, m1, m2):

    m, _ = h.shape
    n, _ = m1.shape

    ratios = p[:n].reshape((-1, 1))
    real_d = p[n:n + m].reshape((-1, 1))
    imag_d = p[n + m:n + 2 * m].reshape((-1, 1))
    real_v = p[n + 2 * m:n + 2 * m + n * m]
    imag_v = p[n + 2 * m + n * m:]

    v = real_v.reshape((n, m)) + 1j * imag_v.reshape((n, m))
    omega = ratios * m1 - m2
    d = real_d + 1j * imag_d

    h_est = v.T @ omega + d
    err = h_est - h
    rerr = np.real(err)
    ierr = np.imag(err)
    cost = 1 / 2 * np.sum(np.abs(err) ** 2)

    gradient = np.zeros_like(p)
    gradient[:n] = np.sum((rerr @ m1.T) * np.real(v.T) +
                          (ierr @ m1.T) * np.imag(v.T), axis=0)

    gradient[n:n + m] = np.sum(rerr, axis=-1)
    gradient[n + m:n + 2 * m] = np.sum(ierr, axis=-1)
    gradient[n + 2 * m:n + 2 * m + n * m] = (omega @ rerr.T).flatten()
    gradient[n + 2 * m + n * m:] = (omega @ ierr.T).flatten()

    return cost, gradient


def test_grad(func, t0, r=range(10)):
    _, grad = func(t0)
    error = np.zeros_like(t0)
    delta = 0.0005
    gra = []
    for i in r:
        print(f"Evaluating variable {i}")
        error[i] += delta
        val1, _ = func(t0 + error)
        val2, _ = func(t0 - error)
        gra.append((val1 - val2) / (2 * delta))
        error[i] -= delta

    print(np.max(grad[:len(gra)] - gra))
