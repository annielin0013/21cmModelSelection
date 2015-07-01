import numpy as np
import matplotlib.pyplot as plt

npix = 30

mean = np.zeros(npix * npix)
cov = np.identity(npix * npix)

# for i in np.arange(100):
# 	for j in np.arange(3):
# 		if i+j <= 99:
# 			cov[i,i+j] = 0.9
# 		if i-j >= 0:
# 			cov[i,i-j] = 0.9
# #cov[-1,-1] = 1000.
#for i in np.arange(50):
#	cov[i,i] = 5.

result1 = np.random.multivariate_normal(mean, cov, 1)
result1 = result1.reshape((npix, npix))
# for i in np.arange(npix):
# 	result1[i,:] *= i
result2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(result1)))

k_radius = 7
result3 = np.zeros_like(result2)
origin = npix/2
for i in np.arange(npix):
	for j in np.arange(npix):
		displacement = np.sqrt(((i - origin) ** 2) + ((j - origin) ** 2))
		if displacement <= k_radius:
			result3[i, j] = result2[i, j]

result4 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result3)))
print result4

plt.figure(1)
plt.imshow(result1, interpolation='nearest')
plt.title('Random Gaussian Field')

plt.figure(2)
plt.imshow(result2.imag, interpolation='nearest')
plt.title('Fourier Gaussian Field')

plt.figure(3)
plt.imshow(result3.real, interpolation='nearest')
plt.title('Low K Field')

plt.figure(4)
plt.imshow(result4.real, interpolation='nearest')
plt.title('Smoothed Gaussian Field')
plt.show()