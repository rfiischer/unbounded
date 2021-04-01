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


def cost_function3(p, h, n, w):

    rec = p[:n]
    imc = p[n:2 * n]
    red = p[2 * n]
    imd = p[2 * n + 1]

    c = rec + 1j * imc
    d = red + 1j * imd

    s = c @ w
    h_est = s + d
    err = np.abs(h_est) ** 2 - np.abs(h) ** 2

    cost = 1 / 2 * np.sum(err ** 2)

    gradient = np.zeros_like(p)
    gradient[:n] = (2 * red * w + 2 * w * s.real) @ err
    gradient[n:2 * n] = (2 * imd * w + 2 * w * s.imag) @ err
    gradient[2 * n] = (2 * red + 2 * s.real) @ err
    gradient[2 * n + 1] = (2 * imd + 2 * s.imag) @ err

    return cost, gradient


def cost_function4(t, h, x):

    # Get params
    n, _ = x.shape

    # Get real and imaginary parts of Cn
    re = t[0:n]
    im = t[n:2 * n]

    # Get the direct path
    red = t[2 * n]
    imd = t[2 * n + 1]

    # Compute real and imaginary parts of h
    reh = x.T @ re
    imh = x.T @ im

    # Compute usefull vectors
    r_e = (reh + red - h.real)
    i_e = (imh + imd - h.imag)

    # Compute cost
    cost = 1 / 2 * (np.sum(r_e ** 2) + np.sum(i_e ** 2))

    # Compute gradient
    gradient = np.zeros_like(t)
    gradient[0:n] = r_e @ x.T
    gradient[n:2 * n] = i_e @ x.T
    gradient[2 * n] = np.sum(r_e)
    gradient[2 * n + 1] = np.sum(i_e)

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

    print(f"Gradient error: {np.max(grad[r] - gra)}")
    print(f"Numerical grad: \n{np.array(gra)}")
    print(f"Analytical grad: \n{grad[r]}")


def optimize_tap(configs, h, method, a=64):

    # Solve linear system
    configs_inv = np.linalg.inv(configs)
    N = len(h)

    if method == 'linear':
        # Get solution
        c = configs_inv @ h

        # Optimize
        c = c.reshape((-1, 1))
        C = c @ np.conj(c.T)
        R = np.real(C)

        v, e = np.linalg.eig(R)

        order = np.argsort(v)[::-1]
        e = e[:, order]

        solution = e[:, 0]
        solution_truncated = np.sign(solution)

        d = 0

    elif method == 'linear-direct':
        # Get direct path
        d = np.average(h[a:])

        # Get solution
        c = configs_inv @ (h - d)

        # Try to optimize
        solution = (c * np.conj(d)).real
        solution_truncated = np.sign(solution)

    elif method == 'nonlinear-simple-single':
        # Get direct path
        d = np.average(h[a:])

        # Get nonlinear behaviour
        hl = np.zeros_like(h)
        hl[:64] = h[:64] - h[N // 2:N // 2 + 64] - d

        # Get solution
        c = configs_inv @ hl

        # Try to optimize
        solution = (c * np.conj(d)).real
        solution_truncated = np.sign(solution)

    elif method == 'nonlinear-simple-average':
        # Get direct path
        d = np.average(h[a:])

        # Get nonlinear behaviour
        na = h.reshape((4, -1))
        nb = np.zeros(N // 4, dtype=complex)
        nb[:64] = np.average(na[1:, :], axis=0)[:64]
        nb[64:] = np.average(na, axis=0)[64:]
        n = np.tile(nb - d, 4)
        hl = np.zeros_like(h)
        hl[:64] = h[:64] - n[:64] - d

        # Get solution
        c = configs_inv @ hl

        # Try to optimize
        solution = (c * np.conj(d)).real
        solution_truncated = np.sign(solution)

    elif method == 'nonlinear-single':
        # Get direct path
        d = np.average(h[a:])

        # Get nonlinear behaviour
        n = np.tile(h[N // 2:] - d, 2)
        hl = h - n - d

        # Get solution
        c = configs_inv @ hl

        # Try to optimize
        solution = (c * np.conj(d)).real
        solution_truncated = np.sign(solution)

    elif method == 'nonlinear-average':
        # Get direct path
        d = np.average(h[a:])

        # Get nonlinear behaviour
        na = h.reshape((4, -1))
        nb = np.zeros(N // 4, dtype=complex)
        nb[:64] = np.average(na[1:, :], axis=0)[:64]
        nb[64:] = np.average(na, axis=0)[64:]
        n = np.tile(nb - d, 4)
        hl = h - n - d

        # Get solution
        c = configs_inv @ hl

        # Try to optimize
        solution = (c * np.conj(d)).real
        solution_truncated = np.sign(solution)

    else:
        raise ValueError("Wrong optimization method.")

    return c, d, solution_truncated
