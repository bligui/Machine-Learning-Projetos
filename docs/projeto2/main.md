# Projeto II  

# Exploração de Dados

# Regressão

# Implementação dos Modelos

# Avaliação dos Modelos

# Relatório Final


# Projeto de Regressão — Página MkDocs

Este documento contém tudo que você pediu: exploração de dados, geração sintética (se necessário), experimentos de regressão com pelo menos três algoritmos (Linear, Random Forest e KNN), avaliação, gráficos e texto interpretativo pronto para a página MkDocs.

> **Como usar**: salve este arquivo em `docs/projeto-regressao.md` no seu projeto MkDocs. Os scripts Python indicados no documento geram os gráficos em `docs/assets/img/` para que sejam exibidos na página.

---

## Sumário

* [Visão Geral](#visão-geral)
* [1. Exploração dos Dados (EDA)](#1-exploração-dos-dados-eda)
* [2. Geração de Dados Sintéticos (se necessário)](#2-geração-de-dados-sintéticos-se-necessário)
* [3. Engenharia de Features](#3-engenharia-de-features)
* [4. Modelos (seleção e código)](#4-modelos-seleção-e-código)
* [5. Avaliação dos Modelos](#5-avaliação-dos-modelos)
* [6. Comparação e Gráficos](#6-comparação-e-gráficos)
* [7. Interpretação e Relatório Final](#7-interpretação-e-relatório-final)
* [8. Instruções para MkDocs](#8-instruções-para-mkdocs)

---

## Visão Geral

Objetivo: prever a variável contínua `nota` a partir de `posicao` e `data` (transformada em `dias_desde_inicio`).

Fluxo do projeto:

1. Carregar e explorar dados
2. Se poucos registros, gerar dados sintéticos realistas
3. Criar features (dias desde início)
4. Treinar pelo menos 3 modelos de regressão
5. Avaliar com métricas apropriadas (R², RMSE, MAE)
6. Comparar, plotar e interpretar

---

## 1. Exploração dos Dados (EDA)

### Estatísticas descritivas

Exiba as estatísticas básicas e verifique valores faltantes, tipos de dados e distribuição da variável alvo.

```python
# scripts/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/docs/projeto2/RankingT.csv'

df = pd.read_csv(url)
print(df.info())
print(df.describe())
print('Missing values:\n', df.isna().sum())

# Converter data e plotar série temporal da nota
df['data'] = pd.to_datetime(df['data'])
df['dias_desde_inicio'] = (df['data'] - df['data'].min()).dt.days

plt.figure(figsize=(10,4))
plt.scatter(df['dias_desde_inicio'], df['nota'])
plt.xlabel('Dias desde início')
plt.ylabel('Nota')
plt.title('Nota ao longo do tempo')
plt.savefig('docs/assets/img/nota_tempo.png')
```

**O que observar**:

* Se `nota` tem variância razoável. Se for quase constante, prever vai ser difícil.
* Se há outliers (valores de nota impossíveis).
* Se `posicao` e `nota` têm relação aparente (plot scatter).

---

## 2. Geração de Dados Sintéticos (se necessário)

Se o dataset for pequeno (ex.: < 100 linhas) ou sem variabilidade suficiente, gere dados sintéticos para enriquecer a análise.

```python
# scripts/generate_synthetic.py
import pandas as pd
import numpy as np

# carregar original
url = 'https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/docs/projeto2/RankingT.csv'
orig = pd.read_csv(url)
orig['data'] = pd.to_datetime(orig['data'])

# parâmetros para simulação
n_to_create = max(0, 300 - len(orig))  # criar até 300 amostras totais

if n_to_create > 0:
    rng = np.random.RandomState(42)
    dias_min = (orig['data'].min())
    dias_max = (orig['data'].max())

    novos = []
    for _ in range(n_to_create):
        pos = int(rng.uniform(orig['posicao'].min(), orig['posicao'].max()))
        dias = int(rng.uniform(0, (orig['data'].max()-orig['data'].min()).days + 30))
        # gerar nota baseada numa função ruidosa (exemplo)
        nota = max(0, min(10, 9.5 - 0.05*pos + 0.01*dias + rng.normal(0,0.5)))
        data = orig['data'].min() + pd.Timedelta(days=dias)
        novos.append({'posicao': pos, 'data': data, 'nota': nota})

    novos_df = pd.DataFrame(novos)
    # juntar e salvar
    final = pd.concat([orig[['posicao','data','nota']], novos_df], ignore_index=True)
    final.to_csv('data/ranking_extended.csv', index=False)
    print('Synthetic data created. Total rows:', len(final))
else:
    print('No synthetic data needed')
```

> A função de geração acima é um exemplo — ajuste a estrutura para manter realismo.

---

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

## 8. Instruções para MkDocs

1. Coloque este arquivo em `docs/projeto-regressao.md`.
2. Crie a pasta `docs/assets/img/` e `docs/assets/`.
3. Coloque os scripts em `scripts/` (fora da pasta docs) e rode:

```bash
python scripts/eda.py
python scripts/generate_synthetic.py  # opcional
python scripts/compare_models.py
```

4. Atualize `mkdocs.yml` adicionando no `nav`:

```yaml
nav:
  - Home: index.md
  - Projeto Regressão: projeto-regressao.md
```

5. Rode `mkdocs serve` e verifique a página.

---

## Arquivos inclusos sugeridos

* `docs/projeto-regressao.md` (este arquivo)
* `scripts/eda.py`
* `scripts/generate_synthetic.py` (opcional)
* `scripts/compare_models.py`
* `docs/assets/img/` (imagens geradas)
* `docs/assets/model_comparison.csv`

---

Se quiser, eu adapto agora os scripts para gerar imagens com um estilo consistente e criar automaticamente as pastas `docs/assets/img/` antes de salvar (pronto para rodar no seu repositório).
