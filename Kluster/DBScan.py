import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn import datasets

X, _ = datasets.make_moons(n_samples=500, noise=0.1, random_state=42)

# Utilizar o DBScan com parametros padroes (ps=0.5, min_samples=5)
dbscan = DBSCAN()
Y = dbscan.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
