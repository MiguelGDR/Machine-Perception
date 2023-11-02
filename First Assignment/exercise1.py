import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# a) ----------------------------------------------------------------------------
Img = imread('images/umbrellas.jpg') # --> Image values are from [0,1] float

# Creation of the plot
plt.subplot(2,2,1)
plt.imshow(Img)
plt.title('Original Image')


# Dimensions
height, width, channels = Img.shape

# b) ----------------------------------------------------------------------------
# I separate each color
I_red = Img[:,:,0]
I_green = Img[:,:,1]
I_blue = Img[:,:,2]

# I sum all 3 channels together and divide by 3 so I get the "value of intensity"
I_gray = ( (I_red + I_green + I_blue) / 3)
plt.subplot(2,2,2)
plt.imshow(I_gray, cmap="gray")
plt.title('Grayscale')

# c) ----------------------------------------------------------------------------
# Display a specific part of the image using only one channel to have a grayscale image
# I'll use the green channel, and the part of the image will be delimited by 130:260 - height, 240:450 - width
# In the end -> [130:260, 240:450, 1]
I_part = Img[130:260, 240:450, 1]
plt.subplot(2,2,3)
plt.imshow(I_part, cmap="gray")
plt.title('Chop of the Image')

# d) ----------------------------------------------------------------------------
# Replace only a part of the image inverting it
# I'll use the same part as I_part but on all channels
I_part_inv = 1 - Img[130:260, 240:450, :]
h, w, c= I_part_inv.shape
print(h,w,c)
# I save the original image in I_inv
I_inv = Img
# Now I change the part where it is inverted
I_inv[130:260, 240:450, :] = I_part_inv
plt.subplot(2,2,4)
plt.imshow(I_inv)
plt.title('Part of the image inverted')

# Show the plot
plt.show()


# e) ----------------------------------------------------------------------------
I_gray_reduced = I_gray[:,:] * 0.3
print(I_gray[1,1], I_gray_reduced[1,1])

plt.clf()
plt.subplot(1,2,1)
plt.imshow(I_gray, cmap="gray", vmin=0, vmax=1)
plt.title('Grayscale')

plt.subplot(1,2,2)
plt.imshow(I_gray_reduced, cmap="gray", vmin=0, vmax=1)
plt.title('Grayscale reduced')

plt.show()



