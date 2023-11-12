import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

from a2_utils import *
# e)
def gauss(sigma):
    # If sigma == 0.5, I add to make vector larger
    add = 0
    if sigma == 0.5:
        add = 0.5

    kernel = np.zeros(int((2 * (sigma * 3)) + 1 + (2 * add)))

    for i in range(int(-(sigma * 3) + add), int((sigma * 3) + add)):
        kernel[int((sigma * 3) + add) + i] = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-(i * i) / (2 * sigma * sigma))

    # To normalize
    kernel = kernel / np.max(kernel)

    return kernel

I = read_data('assignment2/signal.txt')
k1 = gauss(2)
k2 = np.array([0.1,0.6,0.4])
k3 = cv2.filter2D(src = k1, ddepth= -1, kernel= k2)

# s
plt.subplot(1,4,1)
plt.plot(I)
plt.title('Original')

# (s * k1) * k2
I1 = cv2.filter2D(src = I, ddepth= -1, kernel= k1)
I1 = cv2.filter2D(src = I1, ddepth= -1, kernel= k2)
plt.subplot(1,4,2)
plt.plot(I1)
plt.title('(s * k1) * k2')

# (s* k2) *k1
I2 = cv2.filter2D(src = I, ddepth= -1, kernel= k2)
I2 = cv2.filter2D(src = I2, ddepth= -1, kernel= k1)
plt.subplot(1,4,3)
plt.plot(I2)
plt.title('(s * k2) * k1')

# (s* k2) *k1
I3 = cv2.filter2D(src = I, ddepth= -1, kernel= k3)
plt.subplot(1,4,4)
plt.plot(I3)
plt.title('s * (k2 * k1)')

plt.show()