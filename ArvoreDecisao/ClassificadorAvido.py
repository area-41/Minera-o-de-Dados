from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

# Funcao de variacao de Vizinhos
def varia_k(X, Y):
    for k in range(1, 24, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        previsoes = cross_val_predict(knn, X, Y, cv=10)
        f1score = f1_score(Y, previsoes, average='weighted')
        print('k = ', k, ':', round(f1score, 4))

# Carregar o Banco de Dados
dados = datasets.load_digits()
X, Y, = dados.data, dados.target

print('\nBase de Dados Digits, F1-Score variando K:')
varia_k(X, Y)

# Carregar o outro Banco de Dados
dados = datasets.load_iris()
X, Y, = dados.data, dados.target

print('\nBase de Dados Iris, F1-Score variando K:')
varia_k(X, Y)