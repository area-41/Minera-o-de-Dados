import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Criar os Clusters
def executa_kmeans(X, k):
    # inicializar o algoritmo K-Mena com 4 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)

    # executar o algoritmo e armazenar o grupo de cada elemento Y_grupos
    y_grupos = kmeans.fit_predict(X)

    # Plotar os elementos em seus grupos
    plt.scatter(X[:, 0], X[:, 1], c=y_grupos)
    plt.show()

# Gerar dados para Agrupamento
X, _ = make_blobs(n_samples=200, random_state=42, centers=4)


# Plotar o resultado em 3 grupos
executa_kmeans(X, 3)

# Plotar o resultado em 4 grupos
executa_kmeans(X, 4)

