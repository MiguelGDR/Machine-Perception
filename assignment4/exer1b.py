import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *
from a4_utils import *

def harris_points(I, sigma, threshold):
    # First derivatives
    Ix, Iy = first_derivatives(I,sigma)

    # Values of the matrix
    g_ker = gaussdx(1.6 * sigma)

    c11 = cv2.filter2D((Ix**2), -1, g_ker.reshape(1,-1))
    c11 = cv2.filter2D(c11, -1, g_ker.reshape(-1,1))

    c12 = cv2.filter2D((Ix*Iy), -1, g_ker.reshape(1,-1))
    c12 = cv2.filter2D(c12, -1, g_ker.reshape(-1,1))

    c21 = c12

    c22 = cv2.filter2D((Iy**2), -1, g_ker.reshape(1,-1))
    c22 = cv2.filter2D(c22, -1, g_ker.reshape(-1,1))

    # Determinant
    det = (c11 * c22) - (c12 * c21)

    # Trace
    trace = c11 + c22

    # Point condition
    alpha = 0.06
    c = det - alpha * (trace**2)
    c = np.where(c > threshold, c, 0)

    plt.subplot(1,3,2)
    plt.imshow(c)
    plt.title('c')

    h, w = c.shape
    nms = c.copy()  

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (
                c[i, j] > c[i - 1, j - 1]
                and c[i, j] > c[i - 1, j]
                and c[i, j] > c[i - 1, j + 1]
                and c[i, j] > c[i, j - 1]
                and c[i, j] > c[i, j + 1]
                and c[i, j] > c[i + 1, j - 1]
                and c[i, j] > c[i + 1, j]
                and c[i, j] > c[i + 1, j + 1]
            ):
                nms[i, j] = 1
            else:
                nms[i, j] = 0
    
    plt.subplot(1,3,3)
    plt.imshow(nms)
    plt.title('Non Maxima Supression')

    return nms

# Try with test_points.jpg
I = imread_gray('assignment4/data/graf/graf_a.jpg')

plt.figure()

# I compute the function with the image
I_H = harris_points(I,3,50e-6)


plt.subplot(1,3,1)
plt.imshow(I, cmap='gray')
plt.title('Points Detected')
coord_x, coord_y = np.where(I_H == 1)
plt.scatter(coord_y,coord_x,marker='x', color='red', s=100)

plt.show()



