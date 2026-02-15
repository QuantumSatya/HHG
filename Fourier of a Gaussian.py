import numpy as np
import matplotlib.pyplot as plt

N = 256
x = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x, x)

sigma = 0.05
G = np.exp(-(X**2 + Y**2)/(2*sigma**2))

F = np.fft.fftshift(np.fft.fft2(G))
magnitude = np.abs(F)

plt.imshow(np.log(magnitude + 1))
plt.colorbar()
plt.show()

plt.imshow(G)
plt.colorbar()
plt.show()
