import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, y, sigma1, sigma2):
    return (1 / (2 * np.pi * sigma1 * sigma2)) * np.exp(-(((x * x)/(sigma1 * sigma1)) + ((y * y)/(sigma2 * sigma2))))

x_axis = np.arange(-5, 5, .1)

y_axis = np.arange(-5, 5, .1)

X, Y = np.meshgrid(x_axis, y_axis)

z_axis1 = np.array([[gaussian(i, j, 2, 1) for i in x_axis] for j in y_axis])

z_axis2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(z_axis1)))

plt.contour(X, Y, z_axis1)
plt.contour(X, Y, z_axis2)
plt.title('Fourier Transform in 2D')
plt.show()