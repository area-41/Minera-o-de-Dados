from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# Criacao do conjunto de dados
X, Y = make_classification(n_samples=1000, n_informative=10,
                           n_redundant=10, random_state=42) # usa o random state para usar a mesma base de dados, nao aleatorio

# Classificador Arvore de Decisao
arvore = DecisionTreeClassifier(random_state=42, max_depth=7)

# Classificacao de Dados Originais
arvore.fit(X, Y)
acuracia = arvore.score(X, Y)
print('Numero de atributos originais: ', X.shape[1])
print('Acuracia (dados originais): ', acuracia)

# Aplicacao do PCA
pca = PCA(n_components=10)  # numero de atributos que quer reduzir
X = pca.fit_transform(X, Y)

# Classificacao dos dados PCA
arvore.fit(X, Y)
acuracia = arvore.score(X, Y)
print('\nNumero de atributos PCA: ', X.shape[1])
print('Acuracia (dados PCA):', acuracia)