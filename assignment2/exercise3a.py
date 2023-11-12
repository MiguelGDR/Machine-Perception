import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

from UZ_utils import *
from a2_utils import *

# a)
# Recycled code from previous assignment
def myhist(I, bins):
    H = np.zeros(bins) # Creation of the array that will contain the values of the histogram
    I.reshape(-1) # Transform I into a 1D array
    
    # Max and min values in I
    Imax = np.max(I)
    Imin = np.min(I)

    print(Imin, Imax)

    lim = (Imax - Imin)/bins

    for x in range(bins):
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                lim_inf = (x*lim) + Imin
                lim_sup = (x+1) * lim + Imin
                
                if (lim_inf <= I[i,j] and I[i,j] < lim_sup):
                    H[x] = H[x] + 1
        
    return H


def myhist3(I, bins):
    # Empty 3D matrix
    H = np.zeros((3,bins))

    H[0,:] = myhist(I[:,:,0], bins)
    H[1,:] = myhist(I[:,:,1], bins)
    H[2,:] = myhist(I[:,:,2], bins)

    H = H / (I.shape[0] * I.shape[1])

    return H

