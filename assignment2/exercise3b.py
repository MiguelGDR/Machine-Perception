import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

from UZ_utils import *
from a2_utils import *

# b)
def compare_histograms(H1,H2,selection):

    if selection == 0:
        # Euclidean distance
        H = (H1 - H2) * (H1 - H2)
        L = np.sqrt(np.sum(H, axis= 1))
    elif selection == 1:
        # Chi-square
        H = ((H1 - H2) * (H1 - H2)) / (H1 + H2 + 1e-10)
        L = (1/2) * np.sum(H, axis= 1)
    elif selection == 2:
        # Intersection
        L[0] = min(H1[0,:],H2[0,:])
        L[1] = min(H1[1,:],H2[1,:])
        L[2] = min(H1[2,:],H2[2,:])
    elif selection == 3:
        # Hellinger
        H = (np.sqrt(H1) - np.sqrt(H2)) * (np.sqrt(H1) - np.sqrt(H2))
        L = np.sqrt((1/2) * np.sum(H, axis= 1))
    
    return L