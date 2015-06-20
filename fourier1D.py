import numpy as np
from matplotlib import pyplot

def gaussian(x, sigma):
    return (1/(np.sqrt(2 * np.pi * sigma * sigma))) * np.exp(-(x * x)/(2 * sigma * sigma))

x_axis = np.arange(-50, 50, .1)

y_axis1 = np.array([gaussian(i, 1) for i in x_axis])

y_axis2 = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(y_axis1)))

pyplot.plot(x_axis, y_axis1)
pyplot.plot(x_axis, y_axis2)
pyplot.title('Fourier Transform in 1D')
pyplot.show()
