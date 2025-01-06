
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Carregar os dados históricos
dados = pd.read_csv('resultados_megasena.csv')

# Preparar os dados
todos_numeros = [num for col in ['N1', 'N2', 'N3', 'N4', 'N5', 'N6'] for num in dados[col]]
frequencia = pd.DataFrame({'Número': range(1, 61)})
frequencia['Frequência'] = frequencia['Número'].apply(lambda x: todos_numeros.count(x))

# Criar combinações e verificar se já ocorreram
combinacoes = []
for _ in range(10000):  # Gerar 10,000 amostras
    combinacao = sorted(np.random.choice(range(1, 61), 6, replace=False))
    combinacoes.append(combinacao)

df_combinacoes = pd.DataFrame(combinacoes, columns=[f'N{i}' for i in range(1, 7)])
df_combinacoes['Freq'] = df_combinacoes.mean(axis=1)  # Exemplo de uma feature

# Treinar o modelo
X = df_combinacoes[['Freq']]
y = np.random.randint(0, 2, len(X))  # Exemplo de variável alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Gerar combinações mais prováveis
df_combinacoes['Probabilidade'] = modelo.predict_proba(X)[:, 1]
print(df_combinacoes.sort_values(by='Probabilidade', ascending=False).head(5))
