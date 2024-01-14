import numpy as np
import matplotlib.pyplot as plt
from a6_utils import *

# Function to calculate and visualize PCA
def compute_and_visualize_dual_pca(data_file):
    # Load data 
    X = np.loadtxt(data_file)

    # Mean
    mean = np.mean(X, axis=0)

    # Center the data
    Xd = X - mean

    print(Xd)

    # Covariance matrix
    cov_matrix2 = (1 / (len(X) - 1)) * np.dot(Xd.T, Xd) 
    cov_matrix = (1 / (len(X) - 1)) * np.dot(Xd, Xd.T) 

    # SVD
    U, S, VT = np.linalg.svd(cov_matrix)

    # Eigenvectors and eigenvalues
    U2 = np.dot(Xd.T,U) 
    S2 = np.sqrt(1 / S * (len(X) - 1))

    eigenvector1 = U2[:,0] * S2[0]
    eigenvector2 = U2[:,1] * S2[1]

    # Project to U
    projected_data = np.dot(Xd.T, U)

    # Transform back 
    reconstructed_data = np.dot(projected_data, U.T).T + mean

    # Visualize
    drawEllipse(mean, cov_matrix2, 1)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label='Reconstructed Data', marker='x')

    plt.quiver(mean[0], mean[1], eigenvector1[0], eigenvector1[1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.quiver(mean[0], mean[1], eigenvector2[0], eigenvector2[1], color='g', angles='xy', scale_units='xy', scale=1)

    # Set axis limits
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)

    plt.show()

    return


compute_and_visualize_dual_pca('assignment6/data/points.txt')