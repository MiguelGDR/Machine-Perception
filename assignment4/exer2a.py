import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *
from previous_functions import *
from a4_utils import *

def find_correspondences(list1, list2):
    # Hellinger Distances
    L = np.zeros((list1.shape[0],2))

    for i in range(list1.shape[0]):
        min_distance = 10000
        index = (0,0)
        for j in range(list2.shape[0]):
            H = (np.sqrt(list1[i,:]) - np.sqrt(list2[j,:])) * (np.sqrt(list1[i,:]) - np.sqrt(list2[j,:]))
            H = np.sqrt((1/2) * np.sum(H))
            
            if H < min_distance:
                min_distance = H
                index = [i,j]
                
        L[i,0] = index[0]
        L[i,1] = index[1]        

    return L

# Read images
I1 = imread_gray('assignment4/data/graf/graf_a_small.jpg')
I2 = imread_gray('assignment4/data/graf/graf_b_small.jpg')

# Compute the points
I_H1 = hessian_points(I1,3,0.001)
I_H2 = hessian_points(I2,3,0.001)
# Coordenates
cy1, cx1 = np.where(I_H1 == 1)
cy2, cx2 = np.where(I_H2 == 1)  

# Lists
listI1 = simple_descriptors(I1,cy1,cx1)
listI2 = simple_descriptors(I2,cy2,cx2)

# Find matches
L = find_correspondences(listI1, listI2)

# Creation of array of points
p1 = np.zeros((L.shape[0],2))
p2 = np.zeros((L.shape[0],2))

for i in range(L.shape[0]):
    id1, id2  = L[i,:]
    id1 = int(id1)
    id2 = int(id2)
    p1[i,0] = cx1[id1]
    p1[i,1] = cy1[id1]
    p2[i,0] = cx2[id2]
    p2[i,1] = cy2[id2]

# Display
display_matches(I1, p1, I2, p2)

