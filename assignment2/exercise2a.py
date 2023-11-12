import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

from UZ_utils import *
from a2_utils import *

# a) gaussfilter
# Recycled code from previous exercise
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

def gaussfilter(I,gauss_sigma):
    # I generate the kernel
    k = gauss(gauss_sigma)
    k_t = k.reshape(1,-1) # Transposed kernel

    I_cn = I

    # Convolutions through rows
    I_cn = np.array([cv2.filter2D(src = I[i,:], ddepth= -1, kernel=k) for i in range(I.shape[0])])

    # Convolution through columns
    I_cn = np.array([cv2.filter2D(src = I_cn[:,j], ddepth= -1, kernel=k) for j in range(I.shape[1])])

    # Transpose 
    I_cn = [[row[i] for row in I_cn] for i in range(len(I_cn[0]))] 

    return  I_cn

Img = imread('assignment2/images/lena.png')
I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)

plt.subplot(2,3,1)
plt.imshow(I_gray, cmap="gray")
plt.title('Original')

# Gauss noise
I_gn = gauss_noise(I_gray)

plt.subplot(2,3,2)
plt.imshow(I_gn, cmap="gray")
plt.title('Gaussian Noise')

# Fix gaussian noise 
I_fgn = gaussfilter(I_gn, 1)
plt.subplot(2,3,5)
plt.imshow(I_fgn, cmap="gray")
plt.title('Filtered Gaussian Noise')

# Salt an Pepper
I_sp = sp_noise(I_gray)

plt.subplot(2,3,3)
plt.imshow(I_sp, cmap="gray")
plt.title('Salt and Pepper')

# Fix salt and pepper noise 
I_fsp = gaussfilter(I_sp, 2)
plt.subplot(2,3,6)
plt.imshow(I_fsp, cmap="gray")
plt.title('Filtered Salt and Pepper')

plt.show()