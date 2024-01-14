import numpy as np
import matplotlib.pyplot as plt
from a6_utils import *

# Function to calculate and visualize PCA
def compute_and_visualize_pca(data):
    # Load data 
    X = np.loadtxt(data)

    # Mean
    mean = np.mean(X, axis=0)

    # Center the data
    Xd = X - mean

    # Covariance matrix
    cov_matrix = (1 / (len(X) - 1)) * np.dot(Xd.T, Xd) 

    # Eigenvalue and eigenvector 
    U, S, VT = np.linalg.svd(cov_matrix)

    # Project to U
    projected_data = np.dot(Xd, U)

    projected_data[:, 0] = 0

    # Transform back to original space
    reconstructed_data = np.dot(projected_data, U.T) + mean

    # Visualize 
    drawEllipse(mean, cov_matrix, 1)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label='Reconstructed Data', marker='x')

    # Draw normalized eigenvectors as lines
    for i in range(len(U)):
        eigen_vector = U[:, i] # Eigenvector is already normalized
        plt.plot([mean[0], mean[0] + eigen_vector[0]], [mean[1], mean[1] + eigen_vector[1]], color=['red', 'green'][i], label=f'Eigenvector {i + 1}')

    plt.legend()

    # Set axis limits
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)

    plt.show()

# Example usage
compute_and_visualize_pca('assignment6/data/points.txt')
