import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *
from previous_functions import *
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

    # Calculate reprojection errors
    reprojection_errors = []
    for i, (x1, y1, x2, y2) in enumerate(points):
        # Transform point using estimated homography
        transformed_point = np.dot(H, np.array([x1, y1, 1]))
        transformed_point /= transformed_point[2]  # Normalize homogeneous coordinates

        # Calculate Euclidean distance as reprojection error
        error = np.sqrt((transformed_point[0] - x2)**2 + (transformed_point[1] - y2)**2)
        reprojection_errors.append(error)

    # Return both homography matrix and reprojection errors
    return H, reprojection_errors

def ransac(matches, k, threshold):
    best_H = None
    best_inliers = []
    best_error = float('inf')

    for _ in range(k):
        # Randomly select minimal set of matches
        random_indices = np.random.choice(matches.shape[0], 4, replace=False)
        random_matches = matches[random_indices]

        # Estimate homography based on the random matches
        H, reprojection_errors = estimate_homography(random_matches)

        # Ensure reprojection_errors is a NumPy array
        reprojection_errors = np.array(reprojection_errors)

        # Determine inliers based on the reprojection error threshold
        inliers = np.where(reprojection_errors < threshold)[0]

        # If the percentage of inliers is large enough, use the entire inlier subset
        if len(inliers) > 0.8 * matches.shape[0]:
            # Estimate a new homography matrix using all inliers
            H, reprojection_errors = estimate_homography(matches[inliers])

            # Calculate the average reprojection error for the solution
            avg_error = np.mean(reprojection_errors)

            # If the error is lower than any before, update the best solution
            if avg_error < best_error:
                best_error = avg_error
                best_H = H
                best_inliers = inliers

    return best_H, best_inliers

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
    I_H1 = hessian_points(I1,3,0.002)
    I_H2 = hessian_points(I2,3,0.002)

    # Coordenates
    cx1, cy1 = np.where(I_H1 == 1)
    cx2, cy2 = np.where(I_H2 == 1) 

    # Lists
    listI1 = simple_descriptors(I1,cx1,cy1)
    listI2 = simple_descriptors(I2,cx2,cy2)

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

# Read images
I1 = imread_gray('assignment4/data/newyork/newyork_a.jpg')
I2 = imread_gray('assignment4/data/newyork/newyork_b.jpg')

# Functions
p1, p2 = find_matches(I1,I2)

# Combine the matched points into a single array
matches = np.hstack((p1, p2))

# Apply RANSAC
best_H, best_inliers = ransac(matches, k=1000, threshold=2.0)


# Check if a valid homography matrix was found
if best_H is not None:
    # Ensure the homography matrix has the correct data type and shape
    best_H = best_H.astype(np.float32)

    # Use the best homography matrix to transform one image to the other
    warped = cv2.warpPerspective(I1, best_H, (I2.shape[1], I2.shape[0]), flags=cv2.INTER_LINEAR)

    # Plot the results
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(I2)
    plt.title("Image B")

    plt.subplot(1, 2, 2)
    plt.imshow(warped, cmap='gray')
    plt.title("Transformed Image")
    plt.show()
else:
    print("No valid homography matrix found.")