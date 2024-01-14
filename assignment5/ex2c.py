from matplotlib import pyplot as plt
import numpy as np
from a5_utils import *

def reprojection_error(F, point1, point2):
    # Calculate the epipolar line from point1 in the second image
    line = np.dot(F.T, np.hstack((point2, 1)))

    # Calculate the perpendicular distance between point1 and the epipolar line
    distance = np.abs(np.dot(line[:2], point1) + line[2]) / np.sqrt(np.sum(line[:2]**2))

    return distance

data = np.loadtxt('assignment5/data/epipolar/house_points.txt')

F_est, l_points, r_points = fundamental_matrix(data)

# Test with specific points p1 and p2
p1 = np.array([85, 233])
p2 = np.array([67, 219])

dist1 = reprojection_error(F_est, p1, p2)

print(dist1)

# Test with house points
points1 = data[:, :2]
points2 = data[:, 2:]

total_error = 0
for i in range(len(points1)):
    error = reprojection_error(F_est, points1[i], points2[i])
    total_error += error

average_error = total_error / len(points1)

print(average_error)