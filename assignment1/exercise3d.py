import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# d) ----------------------------------------------------------------------------
# Create a mask of the image eagle.jpg
Img = imread('images/eagle.jpg') # --> Image values are from [0,1] float

I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)

plt.subplot(1,3,1)
plt.imshow(I_gray, cmap="gray")
plt.title('Grayscale')

threshold = 0.5

I_bin = np.where(I_gray > threshold, 0, 1)

plt.subplot(1,3,2)
plt.imshow(I_bin, cmap="gray")
plt.title('Binary')

# Erosion / Dilation
n = 5
SE = np.ones((n,n))
for x in range(3):
    Img[:,:,x] = I_bin

I_bin = cv2.dilate(Img, SE)
I_bin = cv2.dilate(I_bin, SE)
I_bin = cv2.erode(I_bin, SE)
I_bin = cv2.erode(I_bin, SE)

plt.subplot(1,3,3)
plt.imshow(I_bin, cmap="gray")
plt.title('Binary Eroded/Dilated')


plt.show()

