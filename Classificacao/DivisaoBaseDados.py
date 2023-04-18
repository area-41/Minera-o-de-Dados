from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Carregar a base de dados
X, Y = load_wine(return_X_y=True)

# Criar Classificador de Arvore de Decisao
arvore = DecisionTreeClassifier(random_state=42)

# Calcular Acuracia sem Difisao da Base de Dados
arvore.fit(X, Y)
acuracia = arvore.score(X, Y)
print('Acuracia sem Divisao: ', acuracia) # resultado falso de que esta acertando 100%

# Dividindo a Base de Dados
X_treino, X_teste, Y_treino, Y_teste = \
    train_test_split(X, Y, train_size=0.7, random_state=42)

# Calcular a Acuracia COM a Divisao da Base de Dados
arvore.fit(X_treino, Y_treino)
acuracia = arvore.score(X_teste, Y_teste)
print('\nAcuracia apos a Divisao: ', acuracia)
