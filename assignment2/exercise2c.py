import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from UZ_utils import *
from a2_utils import *

# c)
def simple_median(I,width):
    w = int((width - 1) / 2)
    result = np.zeros(I.shape[0])
    for i in range(w, I.shape[0] - w): 
        window = np.array([I[i + j] for j in range(-w, w+1)])
        window = np.sort(window,axis=0)
        result[i] = window[w]

    return result

def simple_gaussian(I,width):
    w = int((width - 1) / 2)
    result = np.zeros(I.shape[0])
    for i in range(w, I.shape[0] - w): 
        window = np.array([I[i + j] for j in range(-w, w+1)])

        add = 0
        for x in range(window.shape[0]):
            add = add + window[x]

        result[i] = add / window.shape[0]

    return result

I = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
I = I.reshape((I.shape[0], 1))


plt.subplot(1,4,1)
plt.plot(I)
plt.title('Original')
plt.yticks(range(0, 6))

I_sp = sp_noise(I,0.4)
plt.subplot(1,4,2)
plt.plot(I_sp)
plt.title('Corrupted')
plt.yticks(range(0, 6))

I_g = simple_gaussian(I_sp,3)
plt.subplot(1,4,3)
plt.plot(I_g)
plt.title('Gaussian')
plt.yticks(range(0, 6))

I_m = simple_median(I_sp, 5)
plt.subplot(1,4,4)
plt.plot(I_m)
plt.title('Median')
plt.yticks(range(0, 6))

plt.show()