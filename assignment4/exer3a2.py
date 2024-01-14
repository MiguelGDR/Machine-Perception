import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *
from previous_functions import *
from a4_utils import *

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

I1 = imread_gray('assignment4/data/graf/graf_a.jpg')
I2 = imread_gray('assignment4/data/graf/graf_b.jpg')
data = np.loadtxt('assignment4/data/graf/graf.txt')

p1 = np.zeros((4,2))
p2 = np.zeros((4,2))

for i in range(4):
    p1[i,0] = data[i,0]
    p1[i,1] = data[i,1]
    p2[i,0] = data[i,2]
    p2[i,1] = data[i,3]

display_matches(I1,p1,I2,p2)

H = estimate_homography(data)

warped = cv2.warpPerspective(I1, H, (I2.shape[1], I2.shape[0]))

plt.figure()
plt.imshow(warped, cmap='gray')
plt.title("Transformed Image")
plt.show()