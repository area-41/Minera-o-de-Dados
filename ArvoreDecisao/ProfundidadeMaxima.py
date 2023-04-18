from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

# Criar aBase de Dados
dados = datasets.load_digits()
X, Y = dados.data, dados.target
print('Diferentes Profundidades: ')
for profundidade in range(1, 16):
    # Criar a Arvore
    arvore = tree.DecisionTreeClassifier(random_state=42,
                                         criterion='entropy',
                                         max_depth=profundidade)
    previsoes = cross_val_predict(arvore, X, Y, cv=10)
    f1score = f1_score(Y, previsoes, average='weighted')
    print('Profundidade', profundidade, ': ', round(f1score, 4))

