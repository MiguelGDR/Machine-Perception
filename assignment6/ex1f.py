import numpy as np
import matplotlib.pyplot as plt
from a6_utils import *

# Function to calculate and visualize PCA
def compute_and_visualize_pca(data):
    # Load data 
    X = np.loadtxt(data).T

    # Mean
    mean = np.mean(X, axis=1)

    # Center the data
    Xd = X - mean[:, np.newaxis]

    # Covariance matrix
    cov_matrix = (1 / (len(X[0]) - 1)) * np.dot(Xd, Xd.T) 

    # Eigenvalue and eigenvector 
    U, S, VT = np.linalg.svd(cov_matrix)

    # qpoint (6,6)
    qpoint = np.array([6,6])

    # Visualize 
    drawEllipse(mean, cov_matrix, 1)
    plt.scatter(X[0, :], X[1, :], label='Original Data')
    plt.scatter(qpoint[0], qpoint[1])

    # Distance and closest point
    distances = np.sqrt((X[0] - qpoint[0])**2 + (X[1] - qpoint[1])**2)
    index = np.argmin(distances)
    closest_point = X[:, index]
    print("Closest point:")
    print(closest_point)

    X = np.column_stack((X, qpoint))

    # Center the data again
    Xd = X - mean[:, np.newaxis]

    # Project to U
    projected_data = np.dot(Xd.T,U)

    # Remove variation
    U[0,:] = 0

    # Reconstruct
    reconstructed_data = np.dot(projected_data,U) + mean
    X = reconstructed_data.T

    # Distance and closest point
    distances = np.sqrt((X[0] - X[0,5])**2 + (X[1] - X[1,5])**2)
    index = np.argmin(distances)
    closest_point = X[:, index]
    print("Closest point:")
    print(closest_point)

    plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label='Reconstructed Data', marker='x')

    # Set axis limits
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)

    plt.legend()

    plt.show()

# Example usage
compute_and_visualize_pca('assignment6/data/points.txt')
