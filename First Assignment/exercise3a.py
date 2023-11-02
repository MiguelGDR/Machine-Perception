import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# a) ----------------------------------------------------------------------------
# Basic morphological operations on the image marsk.png
I = imread('images/mask.png') # --> Image values are from [0,1] float

n = 5
SE = np.ones((n,n))
I_eroded = cv2.erode(I,SE)
I_dilated = cv2.dilate(I,SE)

plt.subplot(1,3,1)
plt.imshow(I)
plt.title('Normal')

plt.subplot(1,3,2)
plt.imshow(I_eroded)
plt.title('Eroded')

plt.subplot(1,3,3)
plt.imshow(I_dilated)
plt.title('Dilated')

plt.show()