import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangulate(x1, x2, P1, P2):
    u1, v1 = x1
    u2, v2 = x2

    A = np.array([
        [u1 * P1[2, :] - P1[0, :]],
        [v1 * P1[2, :] - P1[1, :]],
        [u2 * P2[2, :] - P2[0, :]],
        [v2 * P2[2, :] - P2[1, :]]
    ])

    # Aplana la matriz para asegurar que tenga la forma correcta
    A = A.reshape((4, -1))

    # Resolver el sistema sobredeterminado usando SVD (o np.linalg.lstsq)
    _, _, V = np.linalg.svd(A)

    # La solución está en la última columna de V (vector propio correspondiente al menor valor singular)
    X_homogeneous = V[-1, :]

    # Normalizar las coordenadas homogéneas para obtener las coordenadas tridimensionales
    X = X_homogeneous / X_homogeneous[-1]

    return X[:3]

def plot_3d_points(points, transform_matrix=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if transform_matrix is not None:
        points = np.dot(points, transform_matrix.T)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    for i, txt in enumerate(range(len(points))):
        ax.text(points[i, 0], points[i, 1], points[i, 2], str(txt), fontsize=8)

    plt.show()


# Load correspondence points and calibration matrices
data = np.loadtxt("assignment5/data/epipolar/house_points.txt")
P1 = np.loadtxt("assignment5/data/epipolar/house1_camera.txt")
P2 = np.loadtxt("assignment5/data/epipolar/house2_camera.txt")

# Triangulate points
triangulated_points = np.zeros((len(data), 3))
for i in range(len(data)):
    x1 = ([data[i, 0], data[i, 1]])
    x2 = ([data[i, 2], data[i, 3]])
    X = triangulate(x1, x2, P1, P2)

    triangulated_points[i, :] = X 

print(triangulated_points)

# Transformation matrix for visualization
T = np.array([[-1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])

# Visualize the 3D points
plot_3d_points(triangulated_points, transform_matrix=T)

