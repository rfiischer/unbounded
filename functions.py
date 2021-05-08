import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_tnc
from sympy import prime


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
    n = (len(t) - 2) // 2

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


def cost_function5(t, h, x):

    # Get params
    n = len(t) // 2

    # Get real and imaginary parts of Cn
    re = t[:n]
    im = t[n:2 * n]

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
    gradient[0:n] = r_e @ x.T
    gradient[n:2 * n] = i_e @ x.T

    return cost, gradient


def compute_features_1(configs, max_dist):

    # Setup
    s = int(np.sqrt(configs.shape[0]))
    configs = configs.reshape((s, s, -1))
    size = int(s + s * max_dist + np.sum([(s - d) * (max_dist + 1) for d in range(1, max_dist + 1)])) + 1
    features = np.zeros((size, configs.shape[-1]))

    # Get linear features
    features[0, :] = 1

    if max_dist >= 0:
        features[1:s + 1, :] = np.sum(configs, axis=0)

        # Feature counter
        fc = s + 1

        # Get second order features
        for d in range(1, max_dist + 1):
            for i in range(s):
                for j in range(s - d):
                    prod = configs[j, i, :] * configs[j + d, i, :]
                    features[fc, :] += prod

                fc += 1

            for i in range(s - d):
                prod = configs[:, i, :] * configs[:, i + d, :]
                features[fc, :] = np.sum(prod, axis=0)
                fc += 1

                for j in range(1, d + 1):
                    prod1 = configs[:-j, i, :] * configs[j:, i + d, :]
                    prod2 = configs[j:, i, :] * configs[:-j, i + d, :]
                    features[fc, :] = np.sum(prod1 + prod2, axis=0)
                    fc += 1

            for i in range(1, d):
                for j in range(s - i):
                    prod1 = configs[:-d, j, :] * configs[d:, j + i, :]
                    prod2 = configs[d:, j, :] * configs[:-d, j + i, :]
                    features[fc, :] = np.sum(prod1 + prod2, axis=0)
                    fc += 1

    return features


def compute_features_2(configs, num_templates, num_unique, template_indexes, table):

    # Setup
    s = int(np.sqrt(configs.shape[0]))
    ss = int(np.sqrt(s))
    n = configs.shape[-1]
    groups = configs.reshape(ss, ss, s, -1)
    size = len(table) + s + 1
    features = np.zeros((size, n))

    # Reshape
    configs = np.zeros((s, s, n))
    for i in range(ss):
        for j in range(ss):
            configs[i * ss:(i + 1) * ss, j * ss:(j + 1) * ss] = groups[:, :, ss * i + j, :]

    # Get bias features
    features[0, :] = 1
    features[1:1 + s, :] = np.sum(groups, axis=(0, 1))

    # Get feature combinations
    combinations = []
    for temp in template_indexes:
        combinations.append(configs[sl(temp[0]):sl(-temp[2]), sl(temp[1]):sl(-temp[3])] *
                            configs[sl(temp[2]):sl(-temp[0]), sl(temp[3]):sl(-temp[1])])

    # Compute features
    for u in range(num_unique):
        for t in range(num_templates):
            features[1 + s + u * num_templates + t, :] = np.sum(combinations[t]
                                                                [table[u * num_templates + t], :],
                                                                axis=0)

    return features


def get_templates(s, ss, max_dist):

    # Get number of groups per row/column
    n_groups = int(s // ss)

    # Fill every group with a prime number
    group_id = np.zeros((s, s), dtype=int)
    for i in range(n_groups):
        for j in range(n_groups):
            group_id[i * ss:(i + 1) * ss, j * ss:(j + 1) * ss] = prime(n_groups * i + j + 1)

    # Compute the coupling templates (horizontal, vertical, diagonal, ...)
    num_templates = 2 * max_dist * (max_dist + 1)
    template_indexes = np.zeros((num_templates, 4), dtype=int)
    counter = 0
    for d in range(1, max_dist + 1):
        for i in range(d + 1):
            template_indexes[counter] = [0, 0, d, i]
            counter += 1

        for i in range(d):
            template_indexes[counter] = [0, 0, d - i - 1, d]
            counter += 1

        for i in range(d):
            template_indexes[counter] = [i + 1, 0, 0, d]
            counter += 1

        for i in range(d - 1):
            template_indexes[counter] = [d, 0, 0, d - i - 1]
            counter += 1

    # Get group combinations with the templates
    combinations = []
    for temp in template_indexes:
        combinations.append(group_id[sl(temp[0]):sl(-temp[2]), sl(temp[1]):sl(-temp[3])] *
                            group_id[sl(temp[2]):sl(-temp[0]), sl(temp[3]):sl(-temp[1])])

    # Get unique group combinations
    unique = []
    for comb in combinations:
        unique.extend(np.unique(comb))

    unique = np.unique(unique)
    num_unique = len(unique)

    # For each template, find where we have distinct groups
    table = []
    for u in unique:
        for comb in combinations:
            table.append(comb == u)

    return num_templates, num_unique, template_indexes, table


def sl(x):
    # Returns None when x == 0
    # We want that the slice a[:-0] return a instead of []
    return x if x != 0 else None


def features_sizes_1(s, max_dist):

    return int(s + s * max_dist + np.sum([(s - d) * (max_dist + 1) for d in range(1, max_dist + 1)])) + 1


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


def optimize_tap(configs, h, method, a=64, dist=3):

    N = len(h)

    if method == 'linear':

        # Solve linear system
        configs_inv = np.linalg.inv(configs)

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

        # Solve linear system
        configs_inv = np.linalg.inv(configs)

        # Get direct path
        d = np.average(h[a:])

        # Get solution
        c = configs_inv @ (h - d)

        # Try to optimize
        solution = (c * np.conj(d)).real
        solution_truncated = np.sign(solution)

    elif method == 'nonlinear-simple-single':

        # Solve linear system
        configs_inv = np.linalg.inv(configs)

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

        # Solve linear system
        configs_inv = np.linalg.inv(configs)

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

        # Solve linear system
        configs_inv = np.linalg.inv(configs)

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

        # Solve linear system
        configs_inv = np.linalg.inv(configs)

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

    elif method == 'second-order-simple':

        # Setup
        features = compute_features_1(configs, dist)
        size = features.shape[0]

        # Solve
        factor = np.max(np.abs(h))
        h0 = h / factor
        t0 = 0.0005 * np.random.randn(2 * size)
        sol, nit, rc = fmin_tnc(lambda t: cost_function5(t, h0, features), t0)

        d = (sol[0] + 1j * sol[size]) * factor
        nl = (sol[1:size] + 1j * sol[size + 1:]) * factor
        c = np.tile(nl[:64], 64)

        # Get simple solution
        solution = (np.tile(c[:64], 64) * np.conj(d)).real
        solution_truncated = np.sign(solution)

    else:
        raise ValueError("Wrong optimization method.")

    return c, d, solution_truncated


def li_features(feature_matrix, tol=1e-10):

    q, r = np.linalg.qr(feature_matrix.T)

    diagr = np.abs(np.diagonal(r))

    return np.sort(np.where(diagr >= tol)[0])
