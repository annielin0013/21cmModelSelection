from numpy.fft import fft2, ifft2, rfft2, irfft2, fftshift, ifftshift
from numpy import array, arange
from math import sqrt, pi, exp
from matplotlib import pyplot

def gaussian(x, sigma):
    return (1/(sqrt(2*pi*pow(sigma, 2))))*exp(-pow(x, 2)/(2*pow(sigma, 2)))

x_axis = arange(-100, 100, .1)

y_axis1 = array([gaussian(i, 1) for i in x_axis])

y_axis2 = ifftshift(fft2(fftshift(y_axis1)))

pyplot.plot(x_axis, y_axis1)
pyplot.plot(x_axis, y_axis2)
pyplot.show()
