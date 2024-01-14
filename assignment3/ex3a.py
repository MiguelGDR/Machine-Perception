import numpy as np
from matplotlib import pyplot as plt

# Creation of the accumulator array:
def incremenet_and_plot(accumulator, x, y):
    h , w = accumulator.shape

    max_rho = int(np.sqrt((h)**2 + (w)**2))

    for i in range(w):
        # Angle
        theta = (i / w) * np.pi - np.pi/2 
        # Rho
        rho = int(x * np.cos(theta) + y * np.sin(theta))
        rho_index = int((rho + max_rho) * h / (max_rho * 2))
        accumulator[rho_index, i] += 1

    return accumulator
    

accumulator = np.zeros((300,300))
accumulator = incremenet_and_plot(accumulator, 10, 10)
plt.subplot(2,2,1)
plt.imshow(accumulator)

accumulator = np.zeros((300,300))
accumulator = incremenet_and_plot(accumulator, 30, 60)
plt.subplot(2,2,2)
plt.imshow(accumulator)

accumulator = np.zeros((300,300))
accumulator = incremenet_and_plot(accumulator, 50, 20)
plt.subplot(2,2,3)
plt.imshow(accumulator)

accumulator = np.zeros((300,300))
accumulator = incremenet_and_plot(accumulator, 80, 90)
plt.subplot(2,2,4)
plt.imshow(accumulator)


plt.show()