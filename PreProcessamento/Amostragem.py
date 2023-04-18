import pandas as pd
from seaborn import load_dataset

dados = load_dataset('penguins')

# Amostragem sem substituicao
print('\n\nAmostragem sem substituicao')
amostra = dados.sample(frac=0.1)  # frac 10 representa 10% dos dados
print(amostra.index.value_counts().head())

# Amostragem COM substituicao
print('\n\nAmostragem COM substituicao')
amostra2 = dados.sample(n=50, replace=True)  # numero de registros 50, pode repetir
print(amostra2.index.value_counts().head())


# Amostra Estratificada
print('\n\nAmostra Estratificada')
print('Contagem de base de dados original:')
contagem = dados['species'].value_counts()
print(contagem)

amostra = pd.DataFrame()
for n in range(len(contagem)):
    especie = contagem.index[n]
    quantidade = int(contagem[n] * 0.1)
    amostra_especie = dados[dados['species'] == especie].sample(n=quantidade)
    amostra = amostra.append(amostra_especie)

print('Contagem de amostra:')
print(amostra['species'].value_counts())