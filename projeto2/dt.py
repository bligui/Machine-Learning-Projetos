
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Carregar dados
df = pd.read_csv('https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/docs/projeto2/RankingT.csv')

# Converter data
df['data'] = pd.to_datetime(df['data'])
data_inicial = df['data'].min()
df['dias_desde_inicio'] = (df['data'] - data_inicial).dt.days

# Remover ID (não ajuda no modelo)
df = df.drop(columns=['id'])

# Definir X e Y
X = df[['posicao', 'dias_desde_inicio']]
y = df['nota']

# Divisão treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criar modelo
modelo = DecisionTreeRegressor(random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("R² (Árvore de Decisão):", round(r2, 4))
