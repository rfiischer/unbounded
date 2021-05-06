import numpy as np
from scipy.io import savemat

from funcs import model, M

n_angles = 3
max_complexity = 0

population = np.zeros((50, n_angles * M, 64, 64))

for user in range(50):
    # Get linear coefficients
    d = model[user, :, 0].reshape(-1, 1)
    c = model[user, :, 1:65]
    central_direction = np.angle(d)

    # For each direction and channel tap compute rate
    for angle in range(n_angles):
        solutions = np.tile(np.sign((c * np.conj(np.exp(1j * 2 * np.pi / n_angles * angle +
                                                        1j * central_direction))).real), (1, 64)).reshape((-1, 64, 64))

        population[user, angle * M:(angle + 1) * M, :, :] = solutions

        print(f"User: {user}, Angle: {angle}")

savemat(".\\model_solution\\population1.mat", {"population": population})
