import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from a6_utils import *

def compute_and_visualize_pca(data):
    # Load data 
    X = np.loadtxt(data)

    # Mean
    mean = np.mean(X, axis=0)

    # Center the data
    Xd = X - mean

    # Covariance matrix
    cov_matrix = (1 / (len(X) - 1)) * np.dot(Xd.T, Xd) 

    # Analytical eigenvalue and eigenvector calculation
    U, S, VT = np.linalg.svd(cov_matrix)

    eigenvalues = S

    # Normalize eigenvalues
    normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)

    # Cumulative sum of normalized eigenvalues
    cumulative_var_explained = np.cumsum(normalized_eigenvalues)

    # Plot the cumulative graph
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_var_explained, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Eigenvectors')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Graph of Normalized Eigenvalues')
    plt.grid(True)
    plt.show()

# Example usage
compute_and_visualize_pca('assignment6/data/points.txt')
