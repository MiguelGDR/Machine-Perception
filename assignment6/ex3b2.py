from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to calculate and visualize PCA
def compute_dual_pca(data):
    # Load data 
    X = data

    # Mean
    mean = np.mean(X, axis=1)

    # Center the data
    Xd = X - mean[:, np.newaxis]

    # Covariance matrix
    cov_matrix = (1 / (len(X[0]) - 1)) * np.dot(Xd.T, Xd) 

    # SVD
    U, S, VT = np.linalg.svd(cov_matrix)

    # Eigenvectors and eigenvalues
    U2 = np.dot(Xd, U)
    S2 = np.sqrt(1 / ((S + 10**-15) * (len(X[0]) - 1)))

    eigenvectors = np.dot(U2, np.diag(S2))

    # Project first image 
    projected_image = np.dot(X[:,0], eigenvectors)

    # Transform back
    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean
    
    image = reconstructed_data.reshape((96, 84))
    plt.subplot(1,2,1)
    image_original = X[:,0].reshape((96,84))
    plt.imshow(image_original,cmap='gray')

    plt.subplot(1,2,2)
    plt.imshow(image,cmap='gray')
    plt.show()

    return

# Rute
rute = 'assignment6/data/faces/1/'

# Matrix to add
all_columns_matrix = []

for i in range(1, 65): 
    image_name = f'{i:03d}.png' 
    image_rute = os.path.join(rute, image_name)

    image = Image.open(image_rute).convert('L')

    # Reshape into a column 
    column = np.array(image).reshape(-1, 1)

    # Add
    all_columns_matrix.append(column)


final_matrix = np.hstack(all_columns_matrix)

compute_dual_pca(final_matrix)
