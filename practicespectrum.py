import numpy as np
import matplotlib.pyplot as plt

a = .08
k = np.arange(20)
def powerspectrum(a, k):
	return np.cos(a * k)

y = powerspectrum(a, k)
print y

plt.plot(k, y)
plt.show()