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
    
    for x in range(w):  
        for y in range(h): 
            if I[y, x] == 1:  
                for i in range(theta_bins):
                    # Angle
                    theta = (i / theta_bins) * np.pi -np.pi/2 
                    # Rho
                    rho = int(x * np.cos(theta) - y * np.sin(theta))
                    rho_index = int((rho + max_rho) * rho_bins / (max_rho * 2))
                    accumulator[rho_index, i] += 1
    
    plt.imshow(accumulator)
    return accumulator

def nonmaxima_suppression_box(ac):
    h , w = ac.shape
    accumulator = ac
    for i in range(1,h-1):
        for j in range(1,w-1):
            if (ac[i,j] < ac[i-1,j-1]) or (ac[i,j] < ac[i-1,j]) or (ac[i,j] < ac[i-1,j+1]) or (ac[i,j] < ac[i,j-1]) or (ac[i,j] < ac[i,j+1]) or (ac[i,j] < ac[i+1,j-1]) or (ac[i,j] < ac[i+1,j]) or (ac[i,j] < ac[i+1,j+1]):
                accumulator[i,j] = 0
    return accumulator

I = imread_gray('assignment3/images/rectangle.png')
I = findedges(I, 1, 0.16)

plt.subplot(1,2,1)
plt.imshow(I)
plt.title('Image')

plt.subplot(1,2,2)
accumulator = hough_find_lines(I,300,300)
plt.imshow(accumulator)

# nonmaxima_suppression_box
ac = nonmaxima_suppression_box(accumulator)
plt.figure()
plt.imshow(ac)

plt.show()
