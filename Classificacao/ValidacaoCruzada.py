import numpy as np
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Carregar a base de dados
X, Y = load_wine(return_X_y=True)

# Criar Classificador de Arvore de Decisao
arvore = DecisionTreeClassifier(random_state=42)

# Dividir a Base de Dados
X_treino, X_teste, Y_treino, Y_teste = \
    train_test_split(X, Y, train_size=0.7, random_state=42)

# Calcular a Acuracia com Base em Treino e Teste
arvore.fit(X_treino, Y_treino)
acuracia = arvore.score(X_teste, Y_teste)
print('\nAcuracia com Divisao: ', acuracia)

# Calcular a Acuracia com Validacao Cruzada
acuracia = np.mean(cross_val_score(arvore, X, Y, cv=10))
print('\nAcuracia com Validacao Cruzada: ', acuracia)
