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


    # Original Image
    plt.subplot(1,7,1)
    image_original = X[:,0].reshape((96,84))
    plt.imshow(image_original,cmap='gray')
    plt.title('Original')

    # Transform back 32
    eigenvectors[32:,:] = 0

    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean

    image = reconstructed_data.reshape((96, 84))

    plt.subplot(1,7,2)
    plt.imshow(image, cmap='gray')
    plt.title('32')

    # Transform back 16
    eigenvectors[16:,:] = 0

    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean
    image = reconstructed_data.reshape((96, 84))

    plt.subplot(1,7,3)
    plt.imshow(image, cmap='gray')
    plt.title('16')

    # Transform back 8
    eigenvectors[8:,:] = 0

    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean
    image = reconstructed_data.reshape((96, 84))

    plt.subplot(1,7,4)
    plt.imshow(image, cmap='gray')
    plt.title('8')

    # Transform back 4
    eigenvectors[4:,:] = 0

    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean
    image = reconstructed_data.reshape((96, 84))

    plt.subplot(1,7,5)
    plt.imshow(image, cmap='gray')
    plt.title('4')

    # Transform back 2
    eigenvectors[2:,:] = 0

    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean
    image = reconstructed_data.reshape((96, 84))

    plt.subplot(1,7,6)
    plt.imshow(image, cmap='gray')
    plt.title('2')

    # Transform back 1
    eigenvectors[1:,:] = 0

    reconstructed_data = np.dot(projected_image, eigenvectors.T) + mean
    image = reconstructed_data.reshape((96, 84))

    plt.subplot(1,7,7)
    plt.imshow(image, cmap='gray')
    plt.title('1')

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
