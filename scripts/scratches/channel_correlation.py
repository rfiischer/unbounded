import numpy as np
import matplotlib.pyplot as plt

# Constants
Nh = 40             # Number of elements in horizontal axis
N = 40 ** 2         # Total number of elements
wavelength = 1      # Wavelength (can be normalized)
ratio = 8           # Ratio between wavelength and element distance d

# x and y coordinates of every element
x = wavelength / ratio * np.tile(np.arange(0, Nh), Nh)
y = wavelength / ratio * np.repeat(np.arange(0, Nh), Nh)

# Position vector represented as a complex number
u = x + 1j * y

# Each line of U is the element u_n repeated in all columns
U = np.tile(u, (N, 1)).T

# The subtraction U - U.T results in u_n - u_m
R = np.sinc(2 * np.abs(U - U.T) / wavelength)

# Compute eigenvalues and eigenvectors
w, v = np.linalg.eig(R)
order = np.argsort(-np.abs(w))
w = w[order]
v = v[:, order]

# Plot eigenvalues
plt.figure()
plt.semilogy(np.abs(w))
plt.ylim([1e-4, 1e2])
plt.ylabel("Eigenvalue")
plt.xlabel("Eigenvalue Number")

# Plot reshaped eigenvectors
plt.figure()
plt.imshow(v[:, 0].real.reshape(40, 40))

# Generate sample channel
variance = 1
h1 = np.random.multivariate_normal([0] * N, variance * R / 2) + \
     1j * np.random.multivariate_normal([0] * N, variance * R / 2)

plt.figure()
plt.scatter(h1.real, h1.imag)

plt.figure()
plt.imshow(np.abs(h1).reshape(40, 40))
