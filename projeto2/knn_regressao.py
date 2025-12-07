import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Carregar dados
df = pd.read_csv(
    'https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/docs/projeto2/RankingT.csv'
)

# Converter data
df['data'] = pd.to_datetime(df['data'])

# Remover ID
df = df.drop(columns=['id'])

# Definir X e y
X = df[['posicao']]
y = df['nota']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criar e treinar modelo
modelo = KNeighborsRegressor(n_neighbors=5)
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Métricas
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Resultados
print("\n===== KNN REGRESSÃO =====")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Exemplo de previsão
nova_posicao = np.array([[5]])
nota_prevista = modelo.predict(nova_posicao)

print("\nPrevisão de Nota (posição 5):", round(nota_prevista[0], 2))
