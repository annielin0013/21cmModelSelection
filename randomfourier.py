import numpy as np
import matplotlib.pyplot as plt

npix = 4

mean = np.zeros(npix * npix)
cov = np.identity(npix * npix)
result1 = np.random.multivariate_normal(mean, cov, 2)
a = result1[0]
b = result1[1]
result1 = np.array([complex(a[i], b[i]) for i in range(npix * npix)])

#result2 = np.array(result1)
#for j in range((npix * npix)/2):
#	result2[j] =result2[-1 - j].conjugate()
#result2 = result2.reshape ((npix, npix))

result2 = result1.reshape((npix, npix))
origin = (npix / 2, npix / 2)
result2[origin] = result2[origin].real

if npix % 2 == 0:
	for i in range(1, npix):
		for j in range(1, npix / 2):
			result2[i, j] = result2[i, npix - j].conjugate()
	result2copy = np.copy(result2)
	for i in range(1, npix / 2):
		for j in range(1, npix / 2):
			result2copy[i, j] = result2copy[npix - i, j]
	for i in range((npix / 2) + 1, npix):
		for j in range(1, npix / 2):
			result2[i, j] = result2[npix - i, j]
	for i in range(1, npix / 2):
		for j in range(1, npix / 2):
			result2[i, j] = result2copy[i, j]
	for i in range(1, npix / 2):
		for j in range (npix / 2, (npix / 2) + 1):
			result2[i, j] = result2[npix - i, j].conjugate()
	for i in range(npix):
		result2[(i, 0)] = result2[(i, 0)].real
	for j in range(npix):
		result2[(0, j)] = result2[(0, j)].real
	result2[0, 1] = result2[0, 3]
	result2[1, 0] = result2[3, 0]
	print result2

#	for i in range(npix):
#		print result2[i]	

#	print 'start working please'
#	for i in range(npix):
#		print np.fft.ifftshift(result2)[i]
#	assert False

#if npix % 2 == 1:

#k_radius = 3
#for i in np.arange(npix):
#	for j in np.arange(npix):
#		displacement = np.sqrt(((i - origin) ** 2) + ((j - origin) ** 2))
#		k_ring

result3 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result2)))
print 'hey'
print result3

plt.figure(1)
plt.imshow(result2.imag, interpolation='nearest')
plt.title('Random Fourier Field')

plt.figure(2)
plt.imshow(result3.real, interpolation='nearest')
plt.title('Configuration Space')
plt.show()