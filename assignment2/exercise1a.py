import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from a2_utils import *
# b)
# Function simple_convolution

def simple_convolution(signal,kernel):
    I = read_data(signal)
    k = read_data(kernel)

    l_k = len(k)
    l_I = len(I)
    conv = np.zeros(l_I)    # Array I will modify and return

    print(l_I,l_k)

    fpos = int((l_k - 1) / 2)    # First position - fpos
    lpos = int((l_k - fpos - 1))    # Last position - lpos

    for i in range(fpos, lpos):
        sum = 0
        for x in range(l_k):
            sum = (k[x] * I[(i - l_k) + x]) + sum

        conv[i] = sum

    return conv


convolution = simple_convolution('assignment2/signal.txt','assignment2/kernel.txt')

plt.imshow(convolution)
plt.show()