import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *
from a4_utils import *

# Implementationof the function: hessianpoints
def hessian_points(I,sigma,threshold):
    # The image must be in grayscale. 
    # Then, I should compute the second derivativSes
    Ixx, Ixy, Iyy = second_derivatives(I,sigma)

    # Then I compute the determinant
    det_H = Ixx * Iyy - (Ixy**2)
    plt.figure()
    plt.subplot(1,3,2)
    plt.imshow(det_H)
    plt.title('Det_H')
    det_H = np.where(det_H > threshold, det_H, 0)

    h, w = det_H.shape
    nms = det_H.copy()  

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (
                det_H[i, j] > det_H[i - 1, j - 1]
                and det_H[i, j] > det_H[i - 1, j]
                and det_H[i, j] > det_H[i - 1, j + 1]
                and det_H[i, j] > det_H[i, j - 1]
                and det_H[i, j] > det_H[i, j + 1]
                and det_H[i, j] > det_H[i + 1, j - 1]
                and det_H[i, j] > det_H[i + 1, j]
                and det_H[i, j] > det_H[i + 1, j + 1]
            ):
                nms[i, j] = 1
            else:
                nms[i, j] = 0

    
    plt.subplot(1,3,3)
    plt.imshow(nms)
    plt.title('Non Maxima Supression')

    return nms

# Try with graf_a.jpg
I = imread_gray('assignment4/data/graf/graf_a.jpg')

plt.figure()

# I compute the function with the image
I_H = hessian_points(I,3,0.004)


plt.subplot(1,3,1)
plt.imshow(I, cmap='gray')
plt.title('Points Detected')
coord_x, coord_y = np.where(I_H == 1)
plt.scatter(coord_y,coord_x,marker='x', color='red', s=50)

plt.show()
