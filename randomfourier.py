import numpy as np
import matplotlib.pyplot as plt

npix = 4
if npix % 2 == 0:
	npix += 1

mean = np.zeros(npix * npix)
cov = np.identity(npix * npix)
result1 = np.random.multivariate_normal(mean, cov, 2)
a = result1[0]
b = result1[1]

result1 = np.array([complex(a[i], b[i]) for i in range(npix * npix)])
result2 = np.array(result1)
for j in range((npix * npix)/2):
	result2[j] =result2[-1 - j].conjugate()
result2 = result2.reshape ((npix, npix))
result2[(npix / 2, npix / 2)] = result2[(npix / 2, npix / 2)].real
#for x in range(npix):
#	result2[(x, 0)] = result2[(x, 0)].real
#for y in range(npix):
#	result2[(0, y)] = result2[(0, y)].real
print result2

#print result2
#result2 = np.fft.fftshift([result2])
#print result2
#result2[0, 0] = result2[0, 0].real
#print result2
#result2 = np.fft.ifftshift([result2])
#print result2
#result2 = result2.reshape((npix, npix))

#result2[(npix / 2)] = result2[(npix / 2)].real

#print np.fft.ifftshift([-1,0,1])
#print np.fft.fftshift([0,1,-1])
#print result2
#print "hey"

result3 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result2)))
print result3

plt.figure(1)
plt.imshow(result2.imag, interpolation='nearest')
plt.title('Random Fourier Field')

plt.figure(2)
plt.imshow(result3.real, interpolation='nearest')
plt.title('Configuration Space')
plt.show()