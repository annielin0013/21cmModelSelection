import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian(x, y, sigma1, sigma2):
   return (1 / (2 * np.pi * sigma1 * sigma2)) * np.exp(-(((x * x)/(sigma1 * sigma1)) + ((y * y)/(sigma2 * sigma2))))

x_axis = np.arange(-5, 5, .02)
y_axis = np.arange(-5, 5, .02)
X, Y = np.meshgrid(x_axis, y_axis)

z_axis1 = np.array([[gaussian(i, j, .05, .05) for i in x_axis] for j in y_axis])
z_axis2 = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(z_axis1)))

fig = plt.figure()
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(X, Y, z_axis1, cmap=cm.jet)
surf = ax.plot_surface(X, Y, z_axis2, cmap=cm.jet)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Fourier Transform in 3D')
plt.show()