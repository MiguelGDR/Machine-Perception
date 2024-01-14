import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
from UZ_utils import *
from ex2_functions import *

# Creation of the accumulator array:
def hough_find_lines(I, theta_bins, rho_bins):
    h, w = I.shape
    max_rho = int(np.sqrt(h**2 + w**2))
    accumulator = np.zeros((rho_bins, theta_bins))
    
    for x in range(h):  
        for y in range(w): 
            if I[x, y] == 1:  
                for i in range(theta_bins):
                    # Angle
                    theta = (i / theta_bins) * np.pi -np.pi/2 
                    # Rho
                    rho = int(x * np.cos(theta) - y * np.sin(theta))
                    rho_index = int((rho + max_rho) * rho_bins / (max_rho * 2))
                    accumulator[rho_index, i] += 1
    
    plt.imshow(accumulator)
    return accumulator

# Example usage:
# Point
I = np.zeros((100, 100))
I[10, 10] = 1
I[10, 20] = 1

plt.figure()
hough_find_lines(I, 300, 300)
plt.title('Synthetic')

# One line
I = imread_gray('assignment3/images/oneline.png')
I = findedges(I, 1, 0.16)
plt.figure()    
hough_find_lines(I, 300, 300)
plt.title('oneline.png')

# Rectangle
I = imread_gray('assignment3/images/rectangle.png')
I = findedges(I, 1, 0.16)
plt.figure()    
hough_find_lines(I, 300, 300)
plt.title('rectangle.png')

plt.show()
