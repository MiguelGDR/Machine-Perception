import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
from UZ_utils import *
from a3_utils import *
from ex2_functions import *

# Creation of the accumulator array:
def hough_find_lines(I, theta_bins, rho_bins):
    h , w = I.shape

    max_rho = int(np.sqrt((h)**2 + (w)**2))
    accumulator = np.zeros((rho_bins, theta_bins))

    for x in range(h):
        for y in range(w):
            if I[x,y] == 1:
                for i in range(theta_bins):
                    # Angle
                    theta = (i / theta_bins) * np.pi -np.pi/2 
                    # Rho
                    rho = int(x * np.cos(theta) - y * np.sin(theta))
                    rho_index = int((rho + max_rho) * rho_bins / (max_rho * 2))
                    accumulator[rho_index, i] += 1

    return accumulator

def search_values(accumulator, threshold, I, theta_bins, rho_bins):
    # Size of the image
    h , w = I.shape
    max_rho = int(np.sqrt((h)**2 + (w)**2))

    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho,theta] >= threshold:
                draw_line((-max_rho + (2 * max_rho * rho / rho_bins) + 1), (theta / theta_bins) * np.pi, h, w)

    return

I = np.zeros((100,100))
I[10,10] = 1
I[10,20] = 1

plt.imshow(I,cmap='gray')
plt.title('Image')

accumulator = hough_find_lines(I, 300, 300)
search_values(accumulator, 2, I, 300, 300)

# One line
plt.figure()
I = imread_gray('assignment3/images/oneline.png')
plt.imshow(I,cmap='gray')
plt.title('oneline.png')
I = findedges(I, 1, 0.16)

accumulator = hough_find_lines(I, 300, 300)
search_values(accumulator, 50, I, 300, 300)

# Rectangle
plt.figure()
I = imread_gray('assignment3/images/rectangle.png')
plt.imshow(I,cmap='gray')
plt.title('rectangle.png')
I = findedges(I, 1, 0.16)

accumulator = hough_find_lines(I, 300, 300)
search_values(accumulator, 115, I, 300, 300)

plt.show()
