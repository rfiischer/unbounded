import numpy as np
from scipy.io import savemat, loadmat

from funcs import model, M, rate

n_angles = 51
max_complexity = 0

solutions = np.zeros((50, n_angles * M, 64, 64))
rates = np.zeros((50, n_angles * M))

for user in range(50):
    # Get linear coefficients
    d = model[user, :, 0].reshape(-1, 1)
    c = model[user, :, 1:65]
    central_direction = np.angle(d)

    # For each direction and channel tap compute rate
    for angle in range(n_angles):
        solution = np.tile(np.sign((c * np.conj(np.exp(1j * 2 * np.pi / n_angles * angle +
                                                       1j * central_direction))).real), (1, 64)).reshape((-1, 64, 64))

        solutions[user, angle * M:(angle + 1) * M, :, :] = solution

        for i in range(M):
            rates[user, angle * M + i], ht, hf = rate(user, solution[i, :].flatten())

        print(f"User: {user}, Angle: {angle}")


optimal = loadmat("model_solution\\solution3_linear.mat")["theta"]
optimal_rate = np.zeros((50, 1))
for i in range(50):
    optimal_rate[i] = rate(i, optimal[:, i])[0]

population = np.zeros((50, 100, 64))
rates = np.concatenate((rates, optimal_rate), axis=1)
order = np.argsort(-rates)
for user in range(50):

    counter = 0

    for i in range(n_angles * M):

        print(f"User: {user}, index: {i}")

        if counter == 100:
            break

        if i == 0:
            population[user, counter, :] = optimal[:64, user]
            counter += 1

        elif rates[user, order[user, i]] != rates[user, order[user, counter - 1]]:
            population[user, counter, :] = solutions[user, order[user, i], 0, :]
            counter += 1

savemat(".\\model_solution\\population3.mat", {"population": population})
