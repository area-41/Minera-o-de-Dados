from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict

# Carregar a base de dados
data = load_wine()
X, Y = data.data, data.target

# Criar Classificador de Arvore de Decisao e a Classificacao
arvore = DecisionTreeClassifier(max_depth=2, random_state=42)
previsoes = cross_val_predict(arvore, X, Y, cv=10)  # com 10 particoes

# Criar Medidas para as Classes
precisao, revocacao, f1score, suporte = \
    precision_recall_fscore_support(Y, previsoes)

for posicao, classe in enumerate(data.target_names):
    print('\nClasse: ', classe)
    print('(suporte: ', suporte[posicao], ')')
    print('-Precisao: ', round(precisao[posicao], 4))
    print('-Revocacao: ', round(revocacao[posicao], 4))
    print('-F1-Score: ', round(f1score[posicao], 4))


# Calcular a Media Simples das Medidas
precisao, revocacao, f1score, _ = \
    precision_recall_fscore_support(Y, previsoes, average='macro')
print('\nGeral (media simples): ')
print('-Precisao: ', round(precisao, 4))
print('-Revocacao: ', round(revocacao, 4))
print('-F1-Score: ', round(f1score, 4))


# Calcular a Media PONDERADA das medidas
precisao, revocacao, f1score, _ = \
    precision_recall_fscore_support(Y, previsoes, average='weighted')
print('\nGeral (media simples): ')
print('-Precisao: ', round(precisao, 4))
print('-Revocacao: ', round(revocacao, 4))
print('-F1-Score: ', round(f1score, 4))
