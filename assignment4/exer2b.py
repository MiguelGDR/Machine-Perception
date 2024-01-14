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

def find_matches(I1,I2):
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
    L1 = find_correspondences(listI1, listI2)
    L2 = find_correspondences(listI2,listI1)

    # Compare and take only same correspendences between both lists
    L = [] # List
    for a in range(L1.shape[0]):
        for b in range(L2.shape[0]):
            if (L1[a,0] == L2[b,1]) and (L1[a,1] == L2[b,0]):
                L.append([L1[a, 0], L1[a, 1]])

    L = np.array(L) # I change list to an array
   
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
    return


# Read images
I1 = imread_gray('assignment4/data/newyork/newyork_a.jpg')
I2 = imread_gray('assignment4/data/newyork/newyork_b.jpg')

I1 = imread_gray('assignment4/data/graf/graf_a_small.jpg')
I2 = imread_gray('assignment4/data/graf/graf_b_small.jpg')

# Functions
find_matches(I1,I2)