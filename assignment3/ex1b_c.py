import numpy as np
from matplotlib import pyplot as plt
import math
import cv2

# b) Function gaussdx
def gaussdx(sigma):
    # Size of the kernel's array
    gdx = np.zeros((2 * (sigma * 3)) + 1)
    i = 0
    for x in range(-sigma * 3, sigma * 3):
        gdx[i] = (-1/np.sqrt(2*math.pi) * (sigma**3)) * x * math.exp(-(x*x) / (2 * sigma * sigma))
        i += 1
    
    # Normalise kernel adding up their absoulte values
    sum_of_values = 0
    for x in gdx:
        sum_of_values += abs(x)

    # Normalized gdx
    gdx_normalized = gdx / sum_of_values

    return gdx_normalized

# Gauss function
def gauss(sigma):
    # If sigma == 0.5, I add to make vector larger
    add = 0
    if sigma == 0.5:
        add = 0.5

    kernel = np.zeros(int((2 * (sigma * 3)) + 1 + (2 * add)))

    for i in range(int(-(sigma * 3) + add), int((sigma * 3) + add)):
        kernel[int((sigma * 3) + add) + i] = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-(i * i) / (2 * sigma * sigma))

    # To normalize
    sum_val = 0
    for x in kernel:
        sum_val += abs(x)

    kernel = kernel / sum_val

    return kernel

# c) Impulse 
impulse = np.zeros((50,50))
impulse[25,25] = 1

g = gauss(4)
gdx = gaussdx(4)

plt.subplot(2,3,1)
plt.imshow(impulse, cmap='gray')
plt.title('Impulse')

# G and Gt
im_g = cv2.filter2D(impulse, -1, g.reshape(1,-1))
im_g_gt = cv2.filter2D(im_g, -1, g.reshape(-1,1))

plt.subplot(2,3,4)
plt.imshow(im_g_gt, cmap='gray')
plt.title('G, Gt')

# G and Dt
im_g_dt = cv2.filter2D(im_g, -1, gdx.reshape(-1,1))
plt.subplot(2,3,2)
plt.imshow(im_g_dt, cmap='gray')
plt.title('G, Dt')

# D and Gt
im_d = cv2.filter2D(impulse, -1, gdx.reshape(1,-1))
im_d_gt = cv2.filter2D(im_d, -1, g.reshape(-1,1))
plt.subplot(2,3,3)
plt.imshow(im_d_gt, cmap='gray')
plt.title('D, Gt')

# Gt and D
im_gt = cv2.filter2D(impulse, -1, g.reshape(-1,1))
im_gt_d = cv2.filter2D(im_gt, -1, gdx.reshape(1,-1))
plt.subplot(2,3,5)
plt.imshow(im_gt_d, cmap='gray')
plt.title('Gt, D')

# Dt and G
im_dt = cv2.filter2D(impulse, -1, gdx.reshape(-1,1))
im_dt_g = cv2.filter2D(im_dt, -1, g.reshape(1,-1))
plt.subplot(2,3,6)
plt.imshow(im_dt_g, cmap='gray')
plt.title('Dt, G')


plt.show()