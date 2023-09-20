import numpy as np

# Definir los parámetros de la distribución multivariable normal
media = [0, 0]  # Media en las dimensiones x e y
covariance_matrix = [[1, 0.5], [0.5, 1]]  # Matriz de covarianza
  # Número de vectores que deseas generar

# Generar varios vectores de puntos multivariables que sigan una distribución normal
vectores = np.random.multivariate_normal(media, covariance_matrix, size=(2, 500))

vec1 = vectores[0]
# Cada fila de 'vectores' representa un vector de puntos multivariables

print("Vectores generados:")
print(vectores)