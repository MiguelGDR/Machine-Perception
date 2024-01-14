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

    # Covariance matrix
    cov_matrix = (1 / (len(X) - 1)) * np.dot(Xd.T, Xd) 

    # SVD
    U, S, VT = np.linalg.svd(cov_matrix)

    # Eigenvectors and eigenvalues
    eigenvectors = U
    eigenvalues = S
    print("Eigenvectors:")
    print(eigenvectors)
    print("Eigenvalues:")
    print(eigenvalues)

    # Visualize
    drawEllipse(mean, cov_matrix, 1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    return


compute_and_visualize_pca('assignment6/data/points.txt')
