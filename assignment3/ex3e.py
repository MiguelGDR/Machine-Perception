import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
from UZ_utils import *
from a3_utils import *
from ex2_functions import *

# Creation of the accumulator array:
def hough_find_lines(I, theta_bins, rho_bins, threshold):
    h , w = I.shape

    max_rho = int(np.sqrt((h)**2 + (w)**2))
    accumulator = np.zeros((rho_bins, theta_bins))

    for x in range(h):
        for y in range(w):
            if I[x,y] >= threshold:
                for i in range(theta_bins):
                    # Angle
                    theta = (i / theta_bins) * np.pi -np.pi/2 
                    # Rho
                    rho = int(x * np.cos(theta) - y * np.sin(theta))
                    rho_index = int((rho + max_rho) * rho_bins / (max_rho * 2))
                    accumulator[rho_index, i] += 1

    return accumulator

def search_10_values(accumulator, I, theta_bins, rho_bins):
    # Size of the image
    h , w = I.shape
    max_rho = int(np.sqrt((h)**2 + (w)**2))

    top_10_lines = np.unravel_index(np.argsort(accumulator, axis=None)[-10:], accumulator.shape)

    for i in range(10):
        rho, theta = top_10_lines[0][i], top_10_lines[1][i]
        draw_line((-max_rho + (2 * max_rho * rho / rho_bins) + 1), (theta / theta_bins) * np.pi, h, w)

    return

# Bricks
I_gray = imread_gray('assignment3/images/bricks.jpg')
plt.subplot(1,3,1)
plt.imshow(I_gray, cmap='gray')
plt.title('Image')

I_edge = findedges(I_gray,1,0.16)
plt.subplot(1,3,2)
plt.imshow(I_edge, cmap='gray')  
plt.title('Edges')
accumulator = hough_find_lines(I_edge,300,300,0.16)

plt.subplot(1,3,3)
I = imread('assignment3/images/bricks.jpg')
plt.imshow(I)
plt.title('Top 10 lines')
search_10_values(accumulator, I_gray, 300, 300)

# Pier
plt.figure()
I_gray = imread_gray('assignment3/images/pier.jpg')
plt.subplot(1,3,1)
plt.imshow(I_gray, cmap='gray')
plt.title('Image')

I_edge = findedges(I_gray,1,0.16)
plt.subplot(1,3,2)
plt.imshow(I_edge, cmap='gray')  
plt.title('Edges')
accumulator = hough_find_lines(I_edge,300,300,0.16)

plt.subplot(1,3,3)
I = imread('assignment3/images/pier.jpg')
plt.imshow(I)
plt.title('Top 10 lines')
search_10_values(accumulator, I_gray, 300, 300)


plt.show()
