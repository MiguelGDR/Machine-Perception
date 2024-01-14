import numpy as np
from matplotlib import pyplot as plt
import math
import cv2

def gauss(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = np.exp(-(x ** 2) / (2 * sigma ** 2))
	k = k / np.sum(k)
	return np.expand_dims(k, 0)


def gaussdx(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = -x * np.exp(-(x ** 2) / (2 * sigma ** 2))
	k /= np.sum(np.abs(k))
	return np.expand_dims(k, 0)

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