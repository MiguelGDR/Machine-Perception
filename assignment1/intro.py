import cv2  # import entire library
import numpy as np # import library as a specifit, shorter name
from matplotlib import pyplot as plt # import a specific module from a library
from PIL import Image

# Tuples and Lists
at = (1,2,3) # Tuples - Inmutable. Con ()
al = [1,2,3] # Lista - Mutable. Con []

# Sets - No ordenado, no se puede cambiar y no indexado. Con {}
a_set = {1,'a',3}
b = set()
b.add(2)
b.add('b')
# print(1 in a_set and 2 in b) # Comprueba que haya un 1 en a_set y un 2 en b

# Diccionarios - Sirven para dar valor a variables en un mismo array
c = {'d' : 15, 'e' : 'f'}
# print(c['d'])

# Bucle for
"""
a = [1,5,4,2,3,4,5,6,7,8,1]
for x in a:
    print(x)

for i, x in enumerate(a): # Para usarlo con indice
    print(i,x)
"""

# Funciones
def funcionsuma(a,b):
    return a + b

print(funcionsuma(20,8))