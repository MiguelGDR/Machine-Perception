import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os

from UZ_utils import *
from a2_utils import *

# e)
def compare_histograms(H1,H2,selection):

    if selection == 0:
        # Euclidean distance
        H = (H1 - H2) * (H1 - H2)
        L = np.sqrt(np.sum(H, axis= 1))
    elif selection == 1:
        # Chi-square
        H = ((H1 - H2) * (H1 - H2)) / (H1 + H2 + 1e-10)
        L = (1/2) * np.sum(H, axis= 1)
    elif selection == 2:
        # Intersection
        L[0] = min(H1[0,:],H2[0,:])
        L[1] = min(H1[1,:],H2[1,:])
        L[2] = min(H1[2,:],H2[2,:])
    elif selection == 3:
        # Hellinger
        H = (np.sqrt(H1) - np.sqrt(H2)) * (np.sqrt(H1) - np.sqrt(H2))
        L = np.sqrt((1/2) * np.sum(H, axis= 1))
    
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
    H = np.zeros((bins,bins,bins))

    size_bin = 256 / bins   # En el caso de 8 bins, el tamaño de cada bin será 32 [0 a 31]

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            R = I[i,j,0]
            G = I[i,j,1]
            B = I[i,j,2]

            # Variable chose bin
            r = 0
            g = 0
            b = 0
            for i in range(bins):
                if ((i * size_bin) <= R) and (R <= (((i + 1) * size_bin) - 1)):
                    r = i
                if ((i * size_bin) <= G) and (G <= (((i + 1) * size_bin) - 1)):
                    g = i
                if ((i * size_bin) <= B) and (B <= (((i + 1) * size_bin) - 1)):
                    b = i
            
            H[r,g,b] += 1;


    return H

def im_retrieval(path, reference, bins):
    # Reference
    Ir = imread(reference)
    Hr = myhist3(Ir, bins)

    files = os.listdir(path)
    LR = LG = LB = np.zeros(120)
    for file in files:
        I = imread("assignment2/dataset/"+file)
        H = myhist3(I,bins)
        L = compare_histograms(Hr, H, 3)
        LR = np.append(LR, L[0])
        #LG = np.append(LG, L[1])
        #LB = np.append(LB, L[2])

    # Only R histogram-distances for CPU saving
    plt.subplot(1,2,1)
    plt.plot(LR[119:239])
    plt.title('Distances R')

    # Sorted
    LR = np.sort(LR)
    plt.subplot(1,2,2)
    plt.plot(LR[119:239])
    plt.title('Distances R - Sorted')
    

    plt.show()
    
    return 0

im_retrieval('assignment2/dataset/', 'assignment2/dataset/object_01_1.png',8)