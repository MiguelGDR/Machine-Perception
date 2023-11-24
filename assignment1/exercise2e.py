import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from UZ_utils import *

# e) ----------------------------------------------------------------------------
# Create a mask of the image bird.jpg with Otsu`s method` threshold 
def myhist(I):
    H = np.zeros(256) # Creation of the array that will contain the values of the histogram
    I.reshape(-1) # Transform I into a 1D array
    
    # Max and min values in I
    Imax = np.max(I)
    Imin = np.min(I)

    lim = (Imax - Imin)/256

    for x in range(256):
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                lim_inf = (x*lim) + Imin
                lim_sup = (x+1) * lim + Imin
                
                if (lim_inf <= I[i,j] and I[i,j] < lim_sup):
                    H[x] = H[x] + 1
        
    return H

# Image
Img = imread('images/bird.jpg') # --> Image values are from [0,1] float
# Change to [0, 255]
Img = Img * 255
Img = Img.astype(np.uint8)
# Gray image
I_gray = ((Img[:,:,0] + Img[:,:,1] + Img[:,:,2]) / 3)

# Otsu's method
# 1. Histrogram of the image
Hist = myhist(I_gray)
# 2. Normalized Hist / Probability of each intensity
Hist_normalized = Hist / (I_gray.shape[0] * I_gray.shape[1])
# 3. Cumulative probability -> Probability + Previous ones
cum_prob = np.cumsum(Hist_normalized)
# 4. Cumulative means -> (Intensity values * Probabilities) + Previous ones
i_values = np.arange(256)
cum_means = np.cumsum(i_values * Hist_normalized)






