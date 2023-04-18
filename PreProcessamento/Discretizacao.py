from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
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
print('Acuracia: ', acuracia)

# Discretizacao
discretizador = KBinsDiscretizer(n_bins=4)
X = discretizador.fit_transform(X)

# Classificacao dos dados discretizados
knn.fit(X, Y)
acuracia = knn.score(X, Y)

print('\nDados Discretizados: ')
print('Acuracia: ', acuracia)