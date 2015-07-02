import numpy as np
import matplotlib.pyplot as plt

npix = 45

mean = np.zeros(npix * npix)
cov = np.identity(npix * npix)
result1 = np.random.multivariate_normal(mean, cov, 2)
a = result1[0]
b = result1[1]
result1 = np.array([complex(a[i], b[i]) for i in range(npix * npix)])

result2 = result1.reshape((npix, npix))
origin = npix / 2
result2[(origin, origin)] = result2[(origin, origin)].real

if npix % 2 == 0:
    result2[npix - 1:0:-1, -1:npix / 2:-1] = result2[1:, 1:npix / 2].conj()
    result2[npix / 2 - 1:0:-1, npix / 2]=result2[npix / 2 + 1:, npix / 2].conj()
    result2[0, 0:npix] = result2[0, 0:npix].real
    result2[0:npix, 0] = result2[0:npix, 0].real
    result2[0, npix / 2 - 1:0:-1] = result2[0, npix / 2 + 1:]
    result2[npix / 2 - 1:0:-1, 0] = result2[npix / 2 + 1:, 0]

if npix % 2 == 1:
    result2[::-1, npix / 2 - 1::-1] = result2[:, npix / 2 + 1:].conj()
    result2[npix / 2 - 1::-1, npix / 2] = result2[npix / 2 + 1:, npix / 2].conj()

k = np.arange(npix / 2 + 1)
k_ring = np.zeros_like(result2)
means = []
for x in k:
    amp2 = []
    for i in np.arange(npix):
        for j in np.arange(npix):
            displacement = np.sqrt(((i - origin) ** 2) + ((j - origin) ** 2))
            if (k[x] - 1) < displacement < (k[x] + 1):
                k_ring[i, j] = result2[i, j]
                amp = (k_ring[i, j].real ** 2) + (k_ring[i, j].imag ** 2)
                amp2.append(amp)
    mean = np.mean(amp2)
    means.append(mean)

#result3 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result2)))

# plt.figure(1)
# plt.imshow(result2.imag, interpolation='nearest')
# plt.title('Random Fourier Field')

# plt.figure(2)
# plt.imshow(result3.real, interpolation='nearest')
# plt.title('Configuration Space')

# plt.figure(3)
# plt.imshow(k_ring.imag, interpolation='nearest')
# plt.title('K Ring')

plt.plot(k, means)
plt.title('Power Spectrum')
plt.show()