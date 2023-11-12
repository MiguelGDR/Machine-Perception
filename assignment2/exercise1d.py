import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math 

from a2_utils import *
# d)
# Function gauss(sigma)

def gauss(sigma):
    # If sigma == 0.5, I add to make vector larger
    add = 0
    if sigma == 0.5:
        add = 0.5

    kernel = np.zeros(int((2 * (sigma * 3)) + 1 + (2 * add)))

    for i in range(int(-(sigma * 3) + add), int((sigma * 3) + add)):
        kernel[int((sigma * 3) + add) + i] = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-(i * i) / (2 * sigma * sigma))

    # To normalize
    #kernel = kernel / np.max(kernel)

    # I did this following code to plot the different kernels without repeting it outside the function
    diferencia = len(kernel) // 2 
    vector = (np.arange(-diferencia, len(kernel) - diferencia))
    lbl = ("α = {}".format(sigma))
    plt.plot(vector, kernel, label = lbl)

    return kernel

kernel05 = gauss(0.5)
kernel1 = gauss(1)
kernel2 = gauss(2)
kernel3 = gauss(3)
kernel4 = gauss(4)



plt.legend()

plt.show()