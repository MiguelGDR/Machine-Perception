import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# e) ----------------------------------------------------------------------------
# Create a mask of the image coins.jpg
Img = imread('images/coins.jpg') # --> Image values are from [0,1] float

I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)

plt.subplot(1,3,1)
plt.imshow(I_gray, cmap="gray")
plt.title('Grayscale')

threshold = 0.9

I_bin = np.where(I_gray > threshold, 0, 1)

# I change the matrix I_bin values tu uint8 from 0 to 255
I_bin = I_bin * 255
I_bin = I_bin.astype(np.uint8)

plt.subplot(1,3,2)
plt.imshow(I_bin, cmap="gray")
plt.title('Binary')

# I use the function cv2.connectedComponentsWithStats 
num_componentes, etiquetas, estadisticas, centroides = cv2.connectedComponentsWithStats(I_bin, connectivity=8)
for i in range(1, num_componentes): # Start from 1 because 0 is the background
    area = estadisticas[i, cv2.CC_STAT_AREA]
    if area > 700:
        pixels_com = (etiquetas == i)
        I_bin[pixels_com] = 0

plt.subplot(1,3,3)
plt.imshow(I_bin, cmap="gray")
plt.title('Without components with area bigger than 700')

plt.show()