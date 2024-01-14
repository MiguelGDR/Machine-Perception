import numpy as np
import matplotlib.pyplot as plt

# Points
A = np.array([3, 4])
B = np.array([3, 6])
C = np.array([7, 6])
D = np.array([6, 4])

# Matrix with the points in each row
points_matrix = np.vstack((A, B, C, D))

# Calculate the mean of the data
mean_point = np.mean(points_matrix, axis=0)

# Center the data by subtracting the mean
centered_data = points_matrix - mean_point

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(centered_data)

# Eigenvalues 
eigenvalues = S 

# Eigenvectors
eigenvectors = U

# Plot the points and eigenvectors
plt.scatter(points_matrix[:, 0], points_matrix[:, 1], label='Data Points')
plt.quiver(mean_point[0], mean_point[1], eigenvectors[0, 0], eigenvectors[0, 1], scale=3)
plt.quiver(mean_point[0], mean_point[1], eigenvectors[1, 0], eigenvectors[1, 1], scale=3)

plt.grid(True)
plt.show()

# Print the eigenvalues and eigenvectors
print("Eigenvalues")
print(eigenvalues)
print("\nEigenvectors")
print(eigenvectors)
