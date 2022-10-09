# Algunas bibliotecas necesarias

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
sns.set()  # for plot styling
import numpy as np


X, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

#RETO1
def mi_algoritmo_kmeans(X, n_clusters, semilla=2):

    k_means = KMeans(n_clusters)
    k_means.fit(X)
    centroides = k_means.cluster_centers_
    etiquetas = k_means.labels_


    return centroides, etiquetas


# Vamos a probar la función en la variable X previamente creada
centroides, etiquetas = mi_algoritmo_kmeans(X, 4)

# Veamos los resultados
plt.scatter(X[:, 0], X[:, 1], c=etiquetas,s=50, cmap='viridis')
plt.show()

#RetoDos
centroides, etiquetas = mi_algoritmo_kmeans(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=etiquetas,s=50, cmap='viridis')
plt.show()

#Reto3
print(X.shape)
e = 100
X_reescala = np.stack((e*X[:, 0], X[:, 1]), axis=1)
plt.scatter(X_reescala[:, 0], X_reescala[:, 1], s=50);
centroides, etiquetas = mi_algoritmo_kmeans(X_reescala, 3)
plt.scatter(X[:, 0], X[:, 1], c=etiquetas,s=50, cmap='viridis')
plt.show()

##Observaciones:
# se agrupan varios rangos en el eje 0 , y los separados al eje se asignan como similares


