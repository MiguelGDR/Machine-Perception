import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# b) ----------------------------------------------------------------------------
Img = imread('images/bird.jpg') # --> Image values are from [0,1] float

I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)
threshold = 0.20

I_bin = np.where(I_gray < threshold, 0, 1)

for x in range(3):
    Img[:,:,x] = I_bin

# Filter settings
n = 5
SE = np.ones((n,n))

# Erosion / Dilation
I_mask = cv2.dilate(Img, SE)
I_mask = cv2.dilate(I_mask, SE)
I_mask = cv2.dilate(I_mask, SE)
I_mask = cv2.erode(I_mask, SE)
I_mask = cv2.erode(I_mask, SE)
I_mask = cv2.erode(I_mask, SE)

plt.clf()
plt.subplot(1,2,1)
plt.imshow(I_bin, cmap="gray")
plt.title('Binary')

plt.subplot(1,2,2)
plt.imshow(I_mask, cmap="gray")
plt.title('Binary after erosion/dilation')

plt.show()