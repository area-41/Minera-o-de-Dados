from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support

# Carregar a Base de Dados
dados = datasets.load_wine()
X, Y, = dados.data, dados.target

lista_medidas = ['gini', 'entropy']

for medida in lista_medidas:
    # Criar arvore
    arvore = tree.DecisionTreeClassifier(random_state=42,
                                         criterion=medida)
    previsoes = cross_val_predict(arvore, X, Y, cv=10)
    precisao, revocacao, f1score, _ = \
        precision_recall_fscore_support(Y, previsoes,
                                        average='weighted')

    print('\nMedida: ', medida.upper())
    print('-Precisao: ', round(precisao, 5))
    print('-Revocacao: ', round(revocacao, 5))
    print('-F1-Score: ', round(f1score, 5))
