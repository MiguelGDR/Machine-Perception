import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *
from a4_utils import *
import random

def estimate_homography(points):

    # A matrix
    A = np.zeros((2 * len(points), 9))
    for i, (x1, y1, x2, y2) in enumerate(points):
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    # U, S, VT
    U, S, VT = np.linalg.svd(A)

    # h vector
    h = VT[-1, :]

    # H matrix
    H = h.reshape((3, 3))
    H /= H[2, 2]

    return H

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
    I_H1 = hessian_points(I1,6,0.0001)
    I_H2 = hessian_points(I2,6,0.0001)

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

    display_matches(I1, p1, I2, p2)

    return p1, p2

def ransac(data, k, threshold):
    best_inliers = []
    best_homography = None

    for _ in range(k):
        # Randomly points
        indices = np.random.choice(len(data), 4)
        subset = data[indices]

        # Estimate the homography matrix
        H = estimate_homography(subset)

        # Determine inliers
        inliers = []
        for i in range(len(data)):
            point1 = np.array([data[i, 0], data[i, 1], 1])
            point2 = np.array([data[i, 2], data[i, 3], 1])
            projected_point = np.dot(H, point1)
            projected_point /= projected_point[2]
            
            error = np.linalg.norm(point2[:2] - projected_point[:2])
            if error < threshold:
                inliers.append(i)

        #  Check if percentage of inliers is good enough
        if len(inliers) / len(data) > 0.5:
            # If good enough, use the entire inlier 
            inliers_data = data[inliers]
            new_H = estimate_homography(inliers_data)

            # If error of new homography is low, use
            warped = cv2.warpPerspective(I1, new_H, (I2.shape[1], I2.shape[0]))
            error_new = np.sum((warped - I2)**2)
            if best_homography is None or error_new < best_homography[0]:
                best_homography = (error_new, new_H)
                best_inliers = inliers

    return best_homography, best_inliers

# Read images
I1 = imread_gray('assignment4/data/newyork/newyork_a.jpg')
I2 = imread_gray('assignment4/data/newyork/newyork_b.jpg')

#I1 = imread_gray('assignment4/data/graf/graf_a_small.jpg')
#I2 = imread_gray('assignment4/data/graf/graf_b_small.jpg')

# Functions
p1, p2 = find_matches(I1, I2)

# Combine point correspondences
data = np.hstack((p1, p2))

# RANSAC
best_homography, inliers = ransac(data, k=1000, threshold=3.0)

# Use the inliers to estimate the final homography matrix
final_homography = estimate_homography(data[inliers])

# Apply the final homography to visualize the result
warped_final = cv2.warpPerspective(I1, final_homography, (I2.shape[1], I2.shape[0]))

# Display the result
plt.figure()
display_matches(I1, p1, I2, p2)
plt.title("Initial Matches")


display_matches(I1, p1[inliers], I2, p2[inliers])
plt.title("RANSAC Inliers")

plt.imshow(warped_final, cmap='gray')
plt.title("Transformed Image (RANSAC)")
plt.show()