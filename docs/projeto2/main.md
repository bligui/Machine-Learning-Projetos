
# Projeto II  

Este projeto apresenta uma análise de regressão aplicada à previsão de notas de filmes, utilizando diferentes algoritmos de Machine Learning. Ao longo do relatório são abordadas a exploração e preparação dos dados, a implementação dos modelos, a avaliação de desempenho e a comparação entre os resultados obtidos.

---


## Sumário

* [Visão Geral](#visão-geral)
* [1. Exploração dos Dados ](#1-exploração-dos-dados)
* [2. Regressões](#regressões)
* [3. Engenharia de Features](#3-engenharia-de-features)
* [4. Modelos (seleção e código)](#4-modelos-seleção-e-código)
* [5. Avaliação dos Modelos](#5-avaliação-dos-modelos)
* [6. Comparação e Gráficos](#6-comparação-e-gráficos)
* [7. Interpretação e Relatório Final](#7-interpretação-e-relatório-final)


---
## Visão Geral

Objetivo: prever a variável contínua `nota` a partir de `posicao` e `data`.

Fluxo do projeto:

1. Carregar e explorar dados
2. Se poucos registros, gerar dados sintéticos realistas
3. Criar features (dias desde início)
4. Treinar pelo menos 3 modelos de regressão
5. Avaliar com métricas apropriadas (R², RMSE, MAE)
6. Comparar e interpretar.

---

## Exploração de Dados

O dataset usado a partir da raspagem do site do IMDB é estruturado como um registro de séries temporais que monitora o desempenho de 20 filmes por meio de suas notas e posições em um ranking diário de um mês.
O objetivo desta análise exploratória é compreender a estrutura dos dados, avaliar sua qualidade e identificar as relações fundamentais entre as variáveis, servindo como base sólida para o desenvolvimento de modelos de Machine Learning.

!!! tip "O dataset é composto por 5 colunas(id, id_filme, nota, posicao e data) e 600 observações."

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/projeto2/exploracao.py"

    ```
=== "Code"

    ```python
    --8<-- "docs/projeto2/exploracao.py"
    ``` 


A média das notas foi de 5,58, indicando que os filmes, em geral, possuem uma avaliação intermediária. A mediana apresentou valor próximo à média, o que sugere uma distribuição relativamente simétrica. 


- O histograma mostra que a maior concentração de dados está entre 4 e 6, indicando que a maioria dos filmes recebe notas nessa faixa.

[histograma]: projeto2/notahist.png "Histograma de Notas"
![histograma de notas](projeto2/notahist.png)

## Regressão Linear:

=== "Code"

    ```python
    --8<-- "docs/projeto2/linear.py"
    ``` 

> RESULTADOS REGRESSÃO LINEAR - TREINO:

R²: 0.889
RMSE: 0.82
MAE: 0.67

> RESULTADOS REGRESSÃO LINEAR - TESTE:

R²: 0.883
RMSE: 0.82
MAE: 0.67


- O R² está muito alto (perto de 1)
- Erro (RMSE e MAE) é baixo
- Treino e teste estão praticamente iguais pode inidicar a não existência de overfitting

O modelo consegue explicar cerca de 88% da variação da nota dos filmes, usando as variáveis escolhidas.

[dispersão]: projeto2/linear.png "Posição vs Nota"
![Grafico de dispersão](projeto2/linear.png)




## Implementação dos Modelos

---

Os modelos escolhidos foram:
1. Decision tree
2. KNN
3. Random Forest

---

=== "Code"

    ```python
    --8<-- "docs/projeto2/random_forest_regressao.py"
    ``` 
=== "Code"

    ```python
    --8<-- "docs/projeto2/knn_regressao.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/projeto2/dt.py"
    ``` 




## Avaliação dos Modelos

!!!  example" Tabela de Comparação de Resultados"


| Modelo                   | R²         | RMSE      | MAE       |
| ------------------------ | ---------- | --------- | --------- |
| Regressão Linear Simples | 0.8830     | 0.820     | 0.670     |
| Árvore de Decisão        | 0.8989     | 0.759     | 0.588     |
| KNN                      | 0.8816     | 0.821     | 0.625     |
| Random Forest            | **0.8997** | **0.756** | **0.586** |



## Relatório Final








## 3. Engenharia de Features

Transforme `data` em `dias_desde_inicio`, remova `id` e outros campos irrelevantes.

```python
# snippet
df['data'] = pd.to_datetime(df['data'])
df['dias_desde_inicio'] = (df['data'] - df['data'].min()).dt.days
df = df.drop(columns=['id'])
X = df[['posicao','dias_desde_inicio']]
y = df['nota']
```

Escalonamento (necessário para KNN e SVR):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 4. Modelos (seleção e código)

Vamos implementar 4 modelos para comparação (você pediu 3, mas 4 dá melhor visão):

* Regressão Linear Múltipla (baseline)
* Árvore de Decisão (regressor)
* KNN Regressor
* Random Forest Regressor

**Arquivo central de experimentos**: `scripts/compare_models.py`

```python
# scripts/compare_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# carregar (trocável para data/ranking_extended.csv se você gerou sintético)
url = 'https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/docs/projeto2/RankingT.csv'

df = pd.read_csv(url)
df['data'] = pd.to_datetime(df['data'])
df['dias_desde_inicio'] = (df['data'] - df['data'].min()).dt.days
df = df.drop(columns=['id'])
X = df[['posicao','dias_desde_inicio']]
y = df['nota']

# Escalonar para KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=6),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({'model': name, 'r2': r2, 'rmse': rmse, 'mae': mae})
    print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

res_df = pd.DataFrame(results).sort_values('r2', ascending=False)
res_df.to_csv('docs/assets/model_comparison.csv', index=False)

# plot de comparação
plt.figure(figsize=(8,5))
plt.bar(res_df['model'], res_df['r2'])
plt.ylabel('R²')
plt.title('Comparação de modelos (R²)')
plt.savefig('docs/assets/img/model_r2_comparison.png', bbox_inches='tight')

# scatter plot: real x previsto para o melhor modelo
best = res_df.iloc[0]['model']
best_model = models[best]
y_best = best_model.predict(X_test)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_best)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Nota Real')
plt.ylabel('Nota Prevista')
plt.title(f'Real vs Previsto - {best}')
plt.savefig('docs/assets/img/real_vs_predicted.png', bbox_inches='tight')
```

---

## 5. Avaliação dos Modelos

### Métricas sugeridas (regressão):

* **R² (determination coefficient)** — interpretação: proporção da variância explicada.
* **RMSE (Root Mean Squared Error)** — erro padrão das previsões, mesma unidade da variável alvo.
* **MAE (Mean Absolute Error)** — erro médio absoluto, menos sensível a outliers.

Incluí no script as três métricas e exportei uma tabela `docs/assets/model_comparison.csv`.

---

## 6. Comparação e Gráficos

Os gráficos gerados estão em `docs/assets/img/`:

* `nota_tempo.png` — scatter nota x dias
* `model_r2_comparison.png` — barra com R² por modelo
* `real_vs_predicted.png` — scatter real vs previsto para o melhor modelo

No MkDocs, adicione essas imagens com `![alt](assets/img/arquivo.png)`.

---

## 7. Interpretação e Relatório Final

### Estrutura sugerida do texto (cada item pode virar um parágrafo):

1. **Resumo do problema**: prever a nota a partir de posição e tempo.
2. **EDA**: descrever variáveis, número de observações, distribuição da nota, presença de outliers e correlações.
3. **Decisão sobre dados sintéticos**: se gerou, justificar a função usada para simulação e checar se a distribuição final é plausível.
4. **Modelos treinados**: listar modelos, hiperparâmetros principais (ex.: `n_estimators=200`, `max_depth=6`).
5. **Resultado quantitativo**: tabela com R², RMSE e MAE; destacar o melhor modelo.
6. **Gráficos**: incluir `real_vs_predicted` e `model_r2_comparison`.
7. **Interpretação**: explicar o que significam os coeficientes (no caso da regressão linear), e interpretar o desempenho (por exemplo, se R²=0.7, dizer que 70% da variância da nota é explicada pelas variáveis usadas).
8. **Limitações e melhorias**: possíveis fontes de melhoras — mais features (reviews, gênero), validação temporal, tuning de hiperparâmetros, seleção de features e ensembles mais sofisticados.

> **Trecho interpretativo pronto (copiar/colar)**:
>
> "O modelo Random Forest foi o que apresentou melhor desempenho, com R² = X. Isso indica que aproximadamente X% da variabilidade da nota é explicada pelas variáveis posição e tempo. O RMSE de Y expressa que, em média, nossa previsão difere da nota real em Y pontos. A regressão linear, por sua vez, serviu como baseline e permitiu interpretar a direção dos efeitos: por exemplo, cada aumento de uma unidade em `posicao` está associado a uma variação média de *b* pontos na nota, mantendo dias constantes."

---
