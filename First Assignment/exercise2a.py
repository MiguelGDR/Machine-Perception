import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# a) ----------------------------------------------------------------------------
# Create a mask of the image bird.jpg with threshold = 0.3
Img = imread('images/bird.jpg') # --> Image values are from [0,1] float

I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)

# First implementation (with FOR sentence)
I_bin_1 = np.copy(I_gray)

threshold = 0.19

for i in range(I_gray.shape[0]):
    for j in range(I_gray.shape[1]):
        
        if I_gray[i,j] < threshold:
            I_bin_1[i,j] = 0
        else:
            I_bin_1[i,j] = 1

print(I_gray.shape, I_bin_1.shape)

plt.subplot(1,3,1)
plt.imshow(I_gray, cmap="gray")
plt.title('Grayscale')

plt.subplot(1,3,2)
plt.imshow(I_bin_1, cmap="gray")
plt.title('Binary 1')

# Second implementation (with np.where(condition, x, y))
I_bin_2 = np.where(I_gray < threshold, 0, 1)

plt.subplot(1,3,3)
plt.imshow(I_bin_2, cmap="gray")
plt.title('Binary 2')

plt.show()




