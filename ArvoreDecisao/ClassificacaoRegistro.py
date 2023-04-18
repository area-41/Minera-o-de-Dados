from sklearn import datasets
from sklearn import tree

# Carregar a Base de Dados
dados = datasets.load_iris()
X, Y, = dados.data, dados.target

# Criar arvore
arvore = tree.DecisionTreeClassifier(max_depth=2, random_state=42)
arvore.fit(X, Y)

# Fazer Classificacao de Registro
registro = [7, 3.2, 4.7, 1.4]
num_classe = arvore.predict([registro])[0]
classe = dados.target_names[num_classe]
print('Classificando Registro: ', registro)
print('Classe: ', classe)

registro2 = [9, 2.1, 2.3, 4.5]
num_classe = arvore.predict([registro2])[0]
classe = dados.target_names[num_classe]
print('Classificando Registro: ', registro2)
print('Classe 2: ', classe)

registro3 = [1, 7.1, 7.3, 7.5]
num_classe = arvore.predict([registro3])[0]
classe = dados.target_names[num_classe]
print('Classificando Registro: ', registro3)
print('Classe 3: ', classe)