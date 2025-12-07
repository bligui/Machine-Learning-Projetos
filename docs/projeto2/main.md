
# Projeto II  

Este projeto apresenta uma análise de regressão aplicada à previsão de notas de filmes, utilizando diferentes algoritmos de Machine Learning. Ao longo do relatório são abordadas a exploração e preparação dos dados, a implementação dos modelos, a avaliação de desempenho e a comparação entre os resultados obtidos.

---


## Sumário

* [Visão Geral](#visão-geral)
* [1. Exploração dos Dados ](#1-exploração-dos-dados)
* [2. Regressão Linear](#2-regressões)
* [3. Implementação dos Modelos](#3-Implementação-dos-Modelos)
* [4. Avaliação dos Modelos](#4-Avaliação-dos-Modelos)
* [5. Relatório Final](#5-Relatório-Final)
* [6. Possíveis melhorias](#6-Possíveis-melhorias)


---
## Visão Geral

Objetivo: prever a variável contínua `nota` a partir de `posicao` e `data`.

Fluxo do projeto:

1. Carregar e explorar dados
2. Se poucos registros, gerar dados sintéticos realistas
3. Criar features (dias desde início)
4. Treinar pelo menos 3 modelos de regressão
5. Avaliar com métricas apropriadas (R², RMSE, MAE)
6. Comparar e interpretar resultados

---

## Exploração de Dados

O conjunto de dados foi obtido por meio de raspagem do site do IMDB e possui uma estrutura semelhante a uma série temporal, monitorando o desempenho de 20 filmes ao longo de um mês, com registros diários de nota e posição no ranking.
O objetivo desta análise exploratória é compreender a estrutura dos dados, avaliar sua qualidade e identificar as relações fundamentais entre as variáveis, servindo como base sólida para o desenvolvimento de modelos de Machine Learning.

!!! tip "O dataset é composto por 5 colunas(id, id_filme, nota, posicao e data) e 600 observações."

=== "Tabela"

    ```python exec="on" html="0"
    --8<-- "docs/projeto2/exploracao.py"

    ```
=== "Code"

    ```python
    --8<-- "docs/projeto2/exploracao.py"
    ``` 


A média das notas foi de 5,58, indicando que os filmes, em geral, possuem uma avaliação intermediária. A mediana apresentou valor próximo à média, o que sugere uma distribuição relativamente simétrica. 


![histograma de notas](nota.png)

- O histograma mostra que a maior concentração de dados está entre 4 e 6, indicando que a maioria dos filmes recebe notas nessa faixa.

## Regressão Linear:

=== "Codígo Regressão "

    ```python
    --8<-- "docs/projeto2/linear.py"
    ``` 

> RESULTADOS REGRESSÃO LINEAR - TREINO:

R²: 0.889, 
RMSE: 0.82, 
MAE: 0.67

> RESULTADOS REGRESSÃO LINEAR - TESTE:

R²: 0.883,
RMSE: 0.82, 
MAE: 0.67


O valor de R² indica que o modelo explica cerca de 88% da variação da nota dos filmes. Os erros (RMSE e MAE) são relativamente baixos, e a proximidade entre os resultados de treino e teste indica ausência de overfitting significativo.


![Grafico de dispersão](linear.png)


## Implementação dos Modelos

---

Os modelos escolhidos foram:

1. Decision tree
2. KNN
3. Random Forest

---

=== "Random Forest"

    ```python
    --8<-- "docs/projeto2/random_forest_regressao.py"
    ``` 
=== "KNN"

    ```python
    --8<-- "docs/projeto2/knn_regressao.py"
    ``` 

=== "Decision tree"

    ```python
    --8<-- "docs/projeto2/dt.py"
    ``` 




## Avaliação dos Modelos



!!!  example "Tabela de Comparação de Resultados"


| Modelo                   | R²         | RMSE      | MAE       |
| ------------------------ | ---------- | --------- | --------- |
| Regressão Linear Simples | 0.8830     | 0.820     | 0.670     |
| Árvore de Decisão        | 0.8989     | 0.759     | 0.588     |
| KNN                      | 0.8816     | 0.821     | 0.625     |
| Random Forest            | **0.8997** | **0.756** | **0.586** |


### Regressão Linear Simples:

A regressão linear simples apresentou um R² de 0.883, o que indica que aproximadamente 88,3% da variação da nota é explicada pela variável posição. Apesar de apresentar um bom desempenho, esse modelo assume uma relação linear entre as variáveis, o que pode limitar sua capacidade de capturar padrões mais complexos ou não lineares presentes nos dados.

Seu RMSE de 0.82 e MAE de 0.67 mostram que, em média, o erro de previsão está em torno de 0.6 a 0.8 pontos na nota.

É um bom modelo base, mas possui limitação em problemas com maior complexidade.


### Árvore de Decisão:

A Árvore de Decisão apresentou um R² de 0.8989, superior ao da regressão linear simples, indicando que o modelo consegue explicar cerca de 89,9% da variação da nota.

Este modelo é capaz de capturar padrões não lineares, pois ele divide o conjunto de dados em regiões, criando regras de decisão com base nos valores de posição.

O RMSE (0.7592) e o MAE (0.5878) são menores que os do modelo linear, indicando maior precisão na previsão.

Isso mostra que a relação entre posição e nota não é puramente linear, e que a árvore consegue se adaptar melhor aos dados.

### K-Nearest Neighbors (KNN):

O modelo KNN apresentou um R² de 0.8816, valor muito próximo ao da regressão linear. Isso indica que ele explica cerca de 88,1% da variação da nota, porém com desempenho ligeiramente inferior à Árvore e ao Random Forest.

Seu RMSE (0.8213) e MAE (0.6250) foram maiores do que os da Árvore e do Random Forest, mostrando que ele comete erros um pouco maiores na previsão.

Isso pode estar relacionado ao fato de o KNN ser altamente sensível aos dados próximos e à forma como a distância entre eles é calculada.

Apesar disso, ele ainda apresenta um bom desempenho geral.




## Relatório Final

Após a implementação e avaliação dos quatro modelos de regressão, foi possível observar que o Random Forest apresentou o melhor desempenho geral, obtendo o maior valor de R² (0.8997) e os menores valores de RMSE (0.7562) e MAE (0.5857).

Isso indica que o modelo é capaz de explicar quase 90% da variação das notas, além de apresentar o menor erro médio nas previsões.

A Árvore de Decisão também apresentou um bom desempenho, superando a regressão linear simples, o que reforça a existência de relações não lineares entre a posição no ranking e a nota dos filmes.

O KNN, embora eficiente, mostrou resultados ligeiramente inferiores aos dois modelos baseados em árvores.

Dessa forma, o Random Forest foi escolhido como o modelo mais adequado para a previsão da nota dos filmes neste projeto.


## Possíveis Melhorias Futuras

Como melhoria futura, pode ser realizada a inclusão de novas variáveis, como gênero do filme, número de votos, orçamento e popularidade, para enriquecer o conjunto de dados.Também é recomendada a otimização dos hiperparâmetros dos modelos, especialmente do Random Forest e do KNN.
Além disso, a ampliação do período de coleta de dados pode tornar o modelo mais robusto e reduzir possíveis vieses temporais. Por fim, podem ser testados algoritmos mais avançados, como Gradient Boosting e Redes Neurais no futuro.











