import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from UZ_utils import *

# b) Sharp an image

def aumentar_nitidez(imagen):
    # Kernel definition in slide 32
    a = np.array([[0,0,0], [0,2,0], [0,0,0]])
    b = np.array([[1,1,1], [1,1,1], [1,1,1]])

    kernel = a - ((1/9) * b)
    
    # Convolution
    nitida = cv2.filter2D(imagen, -1, kernel)

    return nitida

Img = imread('assignment2/images/museum.jpg')
I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)

plt.subplot(1,2,1)
plt.imshow(I_gray, cmap="gray")
plt.title('Original')

I_sharp = aumentar_nitidez(I_gray)
plt.subplot(1,2,2)
plt.imshow(I_sharp, cmap="gray")
plt.title('Sharp')

plt.show()