import cv2
import numpy as np
from matplotlib import pyplot as plt

# Crear datos (cuatro diferentes 'clusters' de puntos de tipo np.float32):
data = np.float32(np.vstack((
    np.random.randint(0, 20, (25, 2)),
    np.random.randint(30, 50, (25, 2)),
    np.random.randint(60, 80, (25, 2)),
    np.random.randint(90, 110, (25, 2))
)))

print(f"Número total de puntos: {len(data)}")
print(f"Número de dimensiones por punto: {len(data[0])}")

# Visualizar los datos originales
plt.scatter(data[:, 0], data[:, 1], s=20, c='b')
plt.title("Datos Originales")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.show()

# K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
ret, label, center = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Separar los datos usando la salida de etiquetas
A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
C = data[label.ravel() == 2]
D = data[label.ravel() == 3]

print(f"Número de puntos en el grupo A: {len(A)}")
print(A)
print("")

print(f"Número de puntos en el grupo B: {len(B)}")
print(B)
print("")

print(f"Número de puntos en el grupo C: {len(C)}")
print(C)
print("")

print(f"Número de puntos en el grupo D: {len(D)}")
print(D)

# Visualizar los clusters resultantes
plt.scatter(A[:, 0], A[:, 1], c='b', label='Grupo A')
plt.scatter(B[:, 0], B[:, 1], c='g', label='Grupo B')
plt.scatter(C[:, 0], C[:, 1], c='c', label='Grupo C')
plt.scatter(D[:, 0], D[:, 1], c='m', label='Grupo D')
plt.scatter(center[:, 0], center[:, 1], s=100, c='r', label='Centros')
plt.title("Clusters Resultantes")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.legend()
plt.show()
