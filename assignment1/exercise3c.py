import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# c) ----------------------------------------------------------------------------
# immask function:
def immask(Img, mask):

    img_masked = Img * mask

    return img_masked

# I read the image
image = imread('images/bird.jpg') # --> Image values are from [0,1] float

# Gray image
Img = np.copy(image)
I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)
# Threshold
threshold = 0.20
I_bin = np.where(I_gray < threshold, 0, 1)

# To polish the mask:
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

img_masked = immask(image, I_mask)

plt.subplot(1,3,1)
plt.imshow(image)
plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(I_mask, cmap="gray")
plt.title('Mask')

plt.subplot(1,3,3)
plt.imshow(img_masked)
plt.title('Image over mask')

plt.show()