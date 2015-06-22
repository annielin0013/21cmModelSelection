import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mean = [0, 0]
cov = [[1, 0], [0, 100]]
X, Y = np.random.multivariate_normal(mean, cov , 10).T

def gaussian(x, y, sigma1, sigma2):
   return (1 / (2 * np.pi * sigma1 * sigma2)) * np.exp(-(((x * x)/(sigma1 * sigma1)) + ((y * y)/(sigma2 * sigma2))))

Z = np.array([[gaussian(i, j, .05, .05) for i in X] for j in Y])

fig = plt.figure()
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title('Random Gaussian Field')
plt.show()