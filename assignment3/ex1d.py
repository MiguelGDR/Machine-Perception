import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
from UZ_utils import *

# Function gaussdx
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

def partial_derivatives(I,sigma):
    # Takes a grayscale image and returns the partial derivatives both in x an y
    # Creation of the kernel
    d = gaussdx(sigma)

    # Convolution
    Ix = cv2.filter2D(I, -1, d.reshape(1,-1))
    Iy = cv2.filter2D(I, -1, d.reshape(-1,1))

    return Ix, Iy

def second_derivatives(I,sigma):
    # Creation of the kernel
    d = gaussdx(sigma)

    # Convolution
    Ix = cv2.filter2D(I, -1, d.reshape(1,-1))
    Iy = cv2.filter2D(I, -1, d.reshape(-1,1))

    # Second Derivatives
    Ixx = cv2.filter2D(Ix, -1, d.reshape(1,-1))
    Ixy = cv2.filter2D(Ix, -1, d.reshape(-1,1))
    Iyy = cv2.filter2D(Iy, -1, d.reshape(-1,1))

    return Ixx, Ixy, Iyy

def gradient_magnitude(I,sigma):
    # Creation of the kernel
    d = gaussdx(sigma)

    # Convolution
    Ix = cv2.filter2D(I, -1, d.reshape(1,-1))
    Iy = cv2.filter2D(I, -1, d.reshape(-1,1))

    I_mag = np.sqrt((Ix**2) + (Iy**2))
    I_dir = np.arctan2(Iy,Ix)

    return I_mag, I_dir

# Reading of the image and transforming it to grayscale
I = imread('assignment3/images/museum.jpg')
I = ((I[:,:,0] + I[:,:,1] + I[:,:,2]) / 3)

plt.subplot(2,4,1)
plt.imshow(I, cmap='gray')
plt.title('Original')

# Obtaining the derivatives
I_p = np.zeros(2)
I_p = partial_derivatives(I, 2)

plt.subplot(2,4,2)
plt.imshow(I_p[0], cmap='gray')
plt.title('I_x')

plt.subplot(2,4,3)
plt.imshow(I_p[1], cmap='gray')
plt.title('I_y')

# Partial sexond derivatives
I_sp = np.zeros(3)
I_sp = second_derivatives(I, 2)

plt.subplot(2,4,5)
plt.imshow(I_sp[0], cmap='gray')
plt.title('I_xx')

plt.subplot(2,4,6)
plt.imshow(I_sp[1], cmap='gray')
plt.title('I_xy')

plt.subplot(2,4,7)
plt.imshow(I_sp[2], cmap='gray')
plt.title('I_yy')

# Magnitud and direction
I_md = np.zeros(2)
I_md = gradient_magnitude(I,2)

plt.subplot(2,4,4)
plt.imshow(I_md[0], cmap='gray')
plt.title('Magnitud')

plt.subplot(2,4,8)
plt.imshow(I_md[1], cmap='gray')
plt.title('Direction')


plt.show()
