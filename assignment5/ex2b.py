from matplotlib import pyplot as plt
import numpy as np
from a5_utils import *

def fundamental_matrix(points):
    left_points = points[:,:2]
    right_points = points[:,2:4]

    resl, Tl = normalize_points(left_points)
    resr, Tr = normalize_points(right_points)

    A = np.zeros((len(points), 9))
    for i in range(len(points)):
        u1, v1, _ = resl[i]
        u2, v2, _ = resr[i]
        A[i] = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1]

    _, _, Vt = np.linalg.svd(A)

    F = Vt[-1].reshape((3, 3))

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))

    # Denormalize the fundamental matrix
    F = np.dot(np.dot(Tr.T, F), Tl)

    return F, left_points, right_points

def epipolar_lines(im1,im2,F,l_points,r_points):
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    # For right image
    plt.subplot(1,2,2)
    plt.imshow(im2)
    for i in range(len(l_points)):
        # Homogeneous Coordinates
        x = (l_points[i,0], l_points[i,1], 1)

        plt.plot(r_points[i,1], r_points[i,0])

        l2 = np.dot(F, x)

        draw_epiline(l2,h2,w2)

    plt.scatter(r_points[:,0], r_points[:,1], color='red', marker='o', s=50)

 # For right image
    plt.subplot(1,2,1)
    plt.imshow(im1)
    for i in range(len(r_points)):
        x = (r_points[i,0], r_points[i,1], 1)

        plt.plot(r_points[i,1], r_points[i,0])

        l1 = np.dot(F.T, x)

        draw_epiline(l1,h1,w1)

    plt.scatter(l_points[:,0], l_points[:,1], color='red', marker='o', s=50)


    return


data = np.loadtxt('assignment5/data/epipolar/house_points.txt')

F_est, l_points, r_points = fundamental_matrix(data)

im1 = cv2.imread('assignment5/data/epipolar/house1.jpg')
im2 = cv2.imread('assignment5/data/epipolar/house2.jpg')

epipolar_lines(im1,im2,F_est,l_points,r_points)

plt.show()


