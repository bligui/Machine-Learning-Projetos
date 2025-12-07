import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("RankingT.csv")

df['data'] = pd.to_datetime(df['data'])
data_inicial = df['data'].min()
df['dias_desde_inicio'] = (df['data'] - data_inicial).dt.days

y=df['nota']
X=df[['posicao']]

#treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression()
modelo.fit(X, y)

# Previsões separadas para treino e teste
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

coeficiente = modelo.coef_[0]

print("\n>> RESULTADOS REGRESSÃO LINEAR - TREINO:")
print(f"R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")

print("\n>> RESULTADOS REGRESSÃO LINEAR - TESTE:")
print(f"R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
