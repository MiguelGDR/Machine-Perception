import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

from UZ_utils import *
from a2_utils import *

# d)
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
    H = np.zeros((3,bins))

    H[0,:] = myhist(I[:,:,0], bins)
    H[1,:] = myhist(I[:,:,1], bins)
    H[2,:] = myhist(I[:,:,2], bins)

    H = H / (I.shape[0] * I.shape[1])

    return H

def im_retrieval(path, bins):
    I1 = imread(path+"1.png")
    I2 = imread(path+"2.png")
    I3 = imread(path+"3.png")
    I4 = imread(path+"4.png")

    # IMAGES
    plt.subplot(4,4,1)
    plt.imshow(I1)
    plt.title('Image 1')

    plt.subplot(4,4,2)
    plt.imshow(I2)
    plt.title('Image 2')

    plt.subplot(4,4,3)
    plt.imshow(I3)
    plt.title('Image 3')

    plt.subplot(4,4,4)
    plt.imshow(I4)
    plt.title('Image 4')

    # Histograms
    ancho_barra = 0.35

    # Object 1
    H1 = myhist3(I1,8)
    plt.subplot(4,4,5)
    data1 = H1[0,:]
    plt.bar(range(len(data1)), data1, width=ancho_barra)   

    plt.subplot(4,4,9)
    data2 = H1[1,:]
    plt.bar(range(len(data2)), data2, width=ancho_barra)

    plt.subplot(4,4,13)
    data3 = H1[2,:]
    plt.bar(range(len(data3)), data3, width=ancho_barra)

    # Object 2
    H2 = myhist3(I2,8)
    plt.subplot(4,4,6)
    data1 = H2[0,:]
    plt.bar(range(len(data1)), data1, width=ancho_barra)

    plt.subplot(4,4,10)
    data2 = H2[1,:]
    plt.bar(range(len(data2)), data2, width=ancho_barra)

    plt.subplot(4,4,14)
    data3 = H2[2,:]
    plt.bar(range(len(data3)), data3, width=ancho_barra)

    # Object 3
    H3 = myhist3(I3,8)
    plt.subplot(4,4,7)
    data1 = H3[0,:]
    plt.bar(range(len(data1)), data1, width=ancho_barra)

    plt.subplot(4,4,11)
    data2 = H3[1,:]
    plt.bar(range(len(data2)), data2, width=ancho_barra)

    plt.subplot(4,4,15)
    data3 = H3[2,:]
    plt.bar(range(len(data3)), data3, width=ancho_barra)

    # Object 3
    H4 = myhist3(I4,8)
    plt.subplot(4,4,8)
    data1 = H4[0,:]
    plt.bar(range(len(data1)), data1, width=ancho_barra)

    plt.subplot(4,4,12)
    data2 = H4[1,:]
    plt.bar(range(len(data2)), data2, width=ancho_barra)

    plt.subplot(4,4,16)
    data3 = H4[2,:]
    plt.bar(range(len(data3)), data3, width=ancho_barra)

    # L2 distances in RGB
    L1 = compare_histograms(H1,H1,3)
    L2 = compare_histograms(H1,H2,3)
    L3 = compare_histograms(H1,H3,3)
    L4 = compare_histograms(H1,H4,3)

    print(L1[0], L1[1], L1[2])
    print(L2[0], L2[1], L2[2])
    print(L3[0], L3[1], L3[2])
    print(L4[0], L4[1], L4[2])

    plt.show()
    
    return 0

im_retrieval('assignment2/dataset/object_01_',2)