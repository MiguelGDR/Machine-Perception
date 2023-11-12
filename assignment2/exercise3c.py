import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

from UZ_utils import *
from a2_utils import *

# b)
def compare_histograms(H1,H2):

    # Euclidean distance
    H = (H1 - H2) * (H1 - H2)
    L = np.sqrt(np.sum(H, axis= 1))

    return L

def myhist(I, bins):
    H = np.zeros(bins) # Creation of the array that will contain the values of the histogram
    I.reshape(-1) # Transform I into a 1D array
    
    # Max and min values in I
    Imax = np.max(I)
    Imin = np.min(I)

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

I1 = imread('assignment2/dataset/object_01_1.png')
I2 = imread('assignment2/dataset/object_02_1.png')
I3 = imread('assignment2/dataset/object_03_1.png')

plt.subplot(4,3,1)
plt.imshow(I1)
plt.title('Image 1')

plt.subplot(4,3,2)
plt.imshow(I2)
plt.title('Image 2')

plt.subplot(4,3,3)
plt.imshow(I3)
plt.title('Image 3')

# Histograms
ancho_barra = 0.35

# Object 1
H1 = myhist3(I1,8)
plt.subplot(4,3,4)
data1 = H1[0,:]
plt.bar(range(len(data1)), data1, width=ancho_barra)   

plt.subplot(4,3,7)
data2 = H1[1,:]
plt.bar(range(len(data2)), data2, width=ancho_barra)

plt.subplot(4,3,10)
data3 = H1[2,:]
plt.bar(range(len(data3)), data3, width=ancho_barra)

# Object 2
H2 = myhist3(I2,8)
plt.subplot(4,3,5)
data1 = H2[0,:]
plt.bar(range(len(data1)), data1, width=ancho_barra)

plt.subplot(4,3,8)
data2 = H2[1,:]
plt.bar(range(len(data2)), data2, width=ancho_barra)

plt.subplot(4,3,11)
data3 = H2[2,:]
plt.bar(range(len(data3)), data3, width=ancho_barra)

# Object 3
H3 = myhist3(I3,8)
plt.subplot(4,3,6)
data1 = H3[0,:]
plt.bar(range(len(data1)), data1, width=ancho_barra)

plt.subplot(4,3,9)
data2 = H3[1,:]
plt.bar(range(len(data2)), data2, width=ancho_barra)

plt.subplot(4,3,12)
data3 = H3[2,:]
plt.bar(range(len(data3)), data3, width=ancho_barra)

# L2 distances
L1 = compare_histograms(H1,H1)
L2 = compare_histograms(H1,H2)
L3 = compare_histograms(H1,H3)

print(L1[0], L1[1], L1[2])
print(L2[0], L2[1], L2[2])
print(L3[0], L3[1], L3[2])
plt.show()