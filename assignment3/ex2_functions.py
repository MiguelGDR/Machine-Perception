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

def findedges(I,sigma,theta):
    I_gm = gradient_magnitude(I, sigma)
    Imag = np.where(I_gm[0] > theta, I_gm[0], 0)
    Idir = abs(I_gm[1]) * (180/math.pi)
    Inonmax = np.copy(Imag)

    for i in range(1, Imag.shape[0] - 1):
        for j in range(1, Imag.shape[1] - 1):
            angulo = Idir[i,j]

            if (0 <= angulo <= 22.5) or (157.5 <= angulo <= 180):
                neighbors = [Imag[i,j-1],Imag[i,j+1]]
            elif 22.5 <= angulo <= 67.5:
                neighbors = [Imag[i-1,j-1], Imag[i+1,j+1]]
            elif 67.5 <= angulo <= 112.5:
                neighbors = [Imag[i,j-1], Imag[i,j+1]]
            elif 112.5 <= angulo <= 157.5:
                neighbors = [Imag[i+1,j-1], Imag[i-1, j+1]]

            if Imag[i,j] < max(neighbors):
                Inonmax[i,j] = 0
    Inonmax = np.where(Inonmax > 0, 1,0)
    return Inonmax

def nonmaxima_suppression_box(ac):
    h , w = ac.shape
    accumulator = ac
    for i in range(1,h-1):
        for j in range(1,w-1):
            if (ac[i,j] < ac[i-1,j-1]) or (ac[i,j] < ac[i-1,j]) or (ac[i,j] < ac[i-1,j+1]) or (ac[i,j] < ac[i,j-1]) or (ac[i,j] < ac[i,j+1]) or (ac[i,j] < ac[i+1,j-1]) or (ac[i,j] < ac[i+1,j]) or (ac[i,j] < ac[i+1,j+1]):
                accumulator[i,j] = 0
    return accumulator