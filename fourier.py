from numpy.fft import fft, fftshift
from numpy import array, arange
from math import sqrt, pi, exp
from matplotlib import pyplot

def gaussian(x, sigma):
    return (1/(sqrt(2*pi*pow(sigma, 2))))*exp(-pow(x, 2)/(2*pow(sigma, 2)))

x_axis = arange(-1000, 1000, 0.1)

y_axis = array([gaussian(i, 1) for i in x_axis])

pyplot.plot(x_axis, y_axis)
pyplot.show()

y_axis = fftshift(fft(y_axis))
for i in range(0, len(y_axis)):
    y_axis[i] = abs(y_axis[i])

pyplot.plot(x_axis, y_axis)
pyplot.show()
