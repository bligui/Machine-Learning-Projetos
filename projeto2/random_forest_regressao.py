import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np



df = pd.read_csv(
    'https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/docs/projeto2/RankingT.csv'
)


# 2. PRÉ-PROCESSAMENTO


df['data'] = pd.to_datetime(df['data'])

data_inicial = df['data'].min()
df['dias_desde_inicio'] = (df['data'] - data_inicial).dt.days

# Remover a coluna ID
df = df.drop(columns=['id'])




X = df[['posicao', 'dias_desde_inicio']]
y = df['nota']




X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)



modelo = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

modelo.fit(X_train, y_train)




y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== RANDOM FOREST REGRESSÃO =====")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# 7. EXEMPLO DE PREVISÃO


# Exemplo: posição 5 e 30 dias desde o início
nova_amostra = np.array([[5, 30]])
nota_prevista = modelo.predict(nova_amostra)

print("\nPrevisão de Nota (posição 5, 30 dias):", round(nota_prevista[0], 2))
