import numpy as np
import matplotlib.pyplot as plt
from a6_utils import *

# Function to calculate and visualize PCA
def compute_and_visualize_pca(data_file):
    # Load data 
    X = np.loadtxt(data_file)

    # Mean
    mean = np.mean(X, axis=0)

    # Center the data
    Xd = X - mean

    print(Xd)

    # Covariance matrix
    cov_matrix = (1 / (len(X) - 1)) * np.dot(Xd.T, Xd) 

    # SVD
    U, S, VT = np.linalg.svd(cov_matrix)

    # Visualize 
    drawEllipse(mean, cov_matrix, 1)
    plt.scatter(X[:, 0], X[:, 1])

    # Eigenvectors and eigenvalues
    eigenvectors = U
    eigenvalues = S

    print(eigenvectors)

    # Plot eigenvectors
    plt.quiver(mean[0], mean[1], eigenvectors[0, 0] * eigenvalues[0], eigenvectors[1, 0] * eigenvalues[0], color='r', angles='xy', scale_units='xy', scale=1)
    plt.quiver(mean[0], mean[1], eigenvectors[0, 1] * eigenvalues[1], eigenvectors[1, 1] * eigenvalues[1], color='g', angles='xy', scale_units='xy', scale=1)

    # Set axis limits
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)

    plt.show()

# Example usage
compute_and_visualize_pca('assignment6/data/points.txt')
