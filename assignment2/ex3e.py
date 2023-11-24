from matplotlib import pyplot as plt
import os

from UZ_utils import *
from a2_utils import *

# e)
def compare_histograms(H1,H2,selection):

    if selection == 0:
        # Euclidean distance
        H = (H1 - H2) * (H1 - H2)
        L = np.sqrt(np.sum(H))
    elif selection == 1:
        # Chi-square
        H = ((H1 - H2) * (H1 - H2)) / (H1 + H2 + 1e-10)
        L = (1/2) * np.sum(H)
    elif selection == 2:
        # Intersection
        L = np.zeros(3)
        L[0] = min(H1[0,:],H2[0,:])
        L[1] = min(H1[1,:],H2[1,:])
        L[2] = min(H1[2,:],H2[2,:])
    elif selection == 3:
        # Hellinger
        H = (np.sqrt(H1) - np.sqrt(H2)) * (np.sqrt(H1) - np.sqrt(H2))
        L = np.sqrt((1/2) * np.sum(H))
    
    return L

def myhist3(I, bins):
    # Empty 3D matrix
    H = np.zeros((bins,bins,bins))

    size_bin = 1 / bins   # En el caso de 8 bins, el tamaño de cada bin será 32 [0 a 31]

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            R = I[i,j,0]
            G = I[i,j,1]
            B = I[i,j,2]

            # Variable chose bin
            r = 0
            g = 0
            b = 0
            for x in range(bins):
                if ((x * size_bin) <= R) and (R <= ((x + 1) * size_bin)):
                    r = x
                if ((x * size_bin) <= G) and (G <= ((x + 1) * size_bin)):
                    g = x
                if ((x * size_bin) <= B) and (B <= ((x + 1) * size_bin)):
                    b = x
            
            H[r,g,b] += 1

    return H

def im_retrieval(path, reference, bins):
    # Reference
    Ir = imread(reference)
    Hr = myhist3(Ir, bins)

    files = os.listdir(path)
    L_array = 0
    for file in files:
        I = imread("assignment2/dataset/"+file)
        H = myhist3(I,bins)
        L = compare_histograms(Hr, H, 0)
        L_array = np.append(L_array, L)

    # Only R histogram-distances for CPU saving
    plt.subplot(1,2,1)
    plt.plot(L_array)
    plt.title('Distances')

    # Sorted
    L_array_sorted = np.sort(L_array)
    plt.subplot(1,2,2)
    plt.plot(L_array_sorted)
    plt.title('Distances Sorted')   

    plt.show()

    plt.subplot(1,6,1)
    plt.imshow(Ir)
    plt.title('Reference')

    index_sorted = np.argsort(L_array)
    top_5_index = index_sorted[1:6]

    k = 2
    for i in top_5_index:
        pic = imread("assignment2/dataset/"+files[i])
        plt.subplot(1,6,k)
        plt.imshow(pic)
        plt.title({files[i]})
        k += 1
    
    plt.show()
    return 

im_retrieval('assignment2/dataset/', 'assignment2/dataset/object_05_4.png',8)