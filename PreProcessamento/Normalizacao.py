from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import datasets

# Base de dados de vinho
dados = datasets.load_wine()
X, Y = dados.data, dados.target

# Classificador KNN
knn = KNeighborsClassifier()

# Classificacao de Dados Originais
knn.fit(X, Y)
acuracia = knn.score(X, Y)

print('Dados originais: ')
print('Valores minimo e maximo:', X.min(), ',', X.max())
print('Acuracia: ', acuracia)


X = preprocessing.normalize(X, axis=0)
# Classificacao dos dados apos selecao
knn.fit(X, Y)
acuracia = knn.score(X, Y)

print('\nDados normalizados: ')
print('Valores minimo e maximo:', X.min(), ',', X.max())
print('Acuracia: ', acuracia)