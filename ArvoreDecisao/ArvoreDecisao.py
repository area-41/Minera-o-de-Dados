from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree

# Carregar a Base de Dados
dados = datasets.load_iris()
X, Y, = dados.data, dados.target

# Criar arvore
arvore = tree.DecisionTreeClassifier(max_depth=2, random_state=42)
arvore.fit(X, Y)

# Imprimir arvore
print(tree.export_text(arvore))

# Arvore como figura
plt.figure(figsize=(18, 8))
tree.plot_tree(arvore,
               feature_names=dados.feature_names,
               class_names=dados.target_names,
               filled=True)
