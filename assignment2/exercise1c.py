import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from a2_utils import *
# c)
# Function simple_convolution including all elements

def simple_convolution(signal,kernel):
    I = read_data(signal)
    k = read_data(kernel)

    l_k = len(k)
    l_I = len(I)

    conv = np.zeros(l_I)    # Array I will modify and return

    fpos = int((l_k - 1) / 2)    # First position - fpos
    lpos = int((l_I - fpos - 1))    # Last position - lpos

# First part: 0 to fpos - 1


# Medium part: fpos to lpos
    for i in range(fpos, lpos):
        sum = 0
        for x in range(l_k):
            sum = (k[x] * I[(i - fpos) + x]) + sum
        conv[i] = sum
    
    return conv

convolution = simple_convolution('assignment2/signal.txt','assignment2/kernel.txt')

I = read_data('assignment2/signal.txt')
k = read_data('assignment2/kernel.txt')
cvconv = cv2.filter2D(src= I , ddepth= -1,kernel= k)

plt.plot(I, label = 'Original')
plt.plot(k, label = 'Kernel')
plt.plot(convolution, label = 'Result')
plt.plot(cvconv, label = 'cv2')

plt.legend()

plt.show()
