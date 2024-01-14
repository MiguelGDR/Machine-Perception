import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('assignment4/data/newyork/newyork_b.jpg')
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Parámetros para la detección de esquinas
esquinas = cv2.goodFeaturesToTrack(imagen_gris, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Convertir las coordenadas de las esquinas a enteros
esquinas = np.int0(esquinas)

print(esquinas.shape)

# Dibujar círculos alrededor de las esquinas detectadas
for esquina in esquinas:
    x, y = esquina.ravel()
    cv2.circle(imagen, (x, y), 3, 255, -1)

# Mostrar la imagen con las esquinas detectadas
cv2.imshow('Esquinas Detectadas', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
