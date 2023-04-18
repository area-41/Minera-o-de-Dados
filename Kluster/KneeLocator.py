import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kneed import KneeLocator

# Criar Dados
X, _ = make_blobs(n_samples=200, random_state=42, centers=4)

# X com numero de grupos (k) e y com as inercias
lista_x = []
lista_y = []

# Criar um laco variando o K
for k in range(2, 11):
    # executar kmeans para k atual
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # Adicionar a Inercia e K as listas
    lista_x.append(k)
    lista_y.append(kmeans.inertia_)

# Localizar o Joelho do Grafico
knee = KneeLocator(lista_x, lista_y, curve='convex', direction='decreasing')

knee.plot_knee()
plt.show()

melhor_k = lista_x[knee.knee-2]
print('Melhor K: ', melhor_k)