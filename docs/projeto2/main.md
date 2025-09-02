# Projeto de Predição de Doenças Cardiovasculares

Este projeto tem como objetivo aplicar técnicas de **Machine Learning** para prever a presença de doenças cardiovasculares a partir de variáveis clínicas. O desenvolvimento segue etapas bem definidas, cada uma com critérios e pontuação, conforme especificado na rubrica do projeto.

---

## Etapas do Projeto

### 1. Exploração dos Dados

#### Descrição do Conjunto de Dados
O dataset utilizado foi obtido no [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction) e reúne informações provenientes de cinco bases distintas do **UCI Machine Learning Repository**.  
Após a remoção de duplicatas, o conjunto final contém **918 observações** e **12 variáveis**, sendo **11 atributos preditores** e **1 variável alvo** (*HeartDisease*).

#### Variáveis
- **Age**: idade do paciente (anos)  
- **Sex**: sexo (M = Masculino, F = Feminino)  
- **ChestPainType**: tipo de dor no peito  
  - TA: Angina Típica  
  - ATA: Angina Atípica  
  - NAP: Dor Não-Anginosa  
  - ASY: Assintomático  
- **RestingBP**: pressão arterial em repouso (mm Hg)  
- **Cholesterol**: colesterol sérico (mg/dl)  
- **FastingBS**: glicemia em jejum (>120 mg/dl = 1, caso contrário = 0)  
- **RestingECG**: resultados do eletrocardiograma em repouso  
- **MaxHR**: frequência cardíaca máxima atingida  
- **ExerciseAngina**: angina induzida por exercício (Y/N)  
- **Oldpeak**: depressão do segmento ST  
- **ST_Slope**: inclinação do segmento ST (Up, Flat, Down)  
- **HeartDisease**: variável alvo (0 = normal, 1 = presença de doença)  

#### Estatísticas Descritivas e Visualizações
- **Idade**: varia entre ~28 e 77 anos, com média em torno de 53 anos.
=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/idadedesc.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/idadedesc.py"
    ```

- **Sexo**: há predominância do sexo masculino no conjunto.
=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/generodesc.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/generodesc.py"
    ```
- **Colesterol**: grande variabilidade, com valores fora da faixa esperada em alguns casos.
=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/coldesc.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/coldesc.py"
    ```
- **Pressão Arterial em Repouso**: média próxima de 130 mm Hg, condizente com casos de hipertensão.

=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/pressaodesc.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/pressaodesc.py"
    ```
- **MaxHR**: varia entre 60 e 202, indicando ampla faixa de condicionamento físico.

=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/maxhrdesc.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/maxhrdesc.py"
    ```
- **Distribuição da Variável Alvo (HeartDisease)**: aproximadamente **55% dos pacientes apresentam diagnóstico positivo**, o que gera uma base relativamente balanceada para treinamento.
=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/heartdesc.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/heartdesc.py"
    ```

#### Conclusões
- **Distribuição da idade** mostra maior concentração entre 45 e 60 anos.  
- **Proporção por sexo** evidencia predominância masculina.  
- **Boxplots de colesterol e pressão arterial** revelam a presença de outliers que devem ser tratados no pré-processamento.
- **Relação entre ChestPainType e HeartDisease** indica que pacientes assintomáticos (ASY) têm maior probabilidade de diagnóstico positivo.
=== "Result"

    ```python exec="on" html="1"
    --8<-- "docs/roteiro1/est/relacao.py"
    ``` 

=== "Code"

    ```python
    --8<-- "docs/roteiro1/est/relacao.py"
    ```

---

### 2. Pré-processamento
#### Limpeza dos Dados

Antes de qualquer análise ou modelagem, é essencial garantir que os dados estejam **consistentes e utilizáveis**.  
Neste passo, foram realizadas duas etapas principais:  

- **Tratamento de valores ausentes**: substituímos os valores nulos nas colunas numéricas pela **mediana**, pois ela é menos sensível a outliers do que a média.  
- **Detecção de outliers**: utilizamos o **Z-Score** para identificar valores que se desviam muito da distribuição normal dos dados. Embora os outliers não tenham sido removidos nesta etapa, sua identificação é fundamental para entender possíveis distorções.  

=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/roteiro1/pre/tratar.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/roteiro1/pre/tratar.py"
    ```

#### Codificação de Variáveis Categóricas

Os algoritmos de Machine Learning trabalham apenas com **valores numéricos**. Portanto, variáveis categóricas como `"Sex"`, `"ChestPainType"`, `"RestingECG"`, `"ExerciseAngina"` e `"ST_Slope"` foram convertidas em números usando **mapeamento direto**.  

Exemplos:  
- `"M"` → `1` e `"F"` → `0`  
- `"ASY"` → `3`, `"NAP"` → `2`, `"ATA"` → `1`, `"TA"` → `0`  

Essa etapa é essencial para que o modelo consiga interpretar as categorias sem perder o significado original.  

=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/roteiro1/pre/encoding.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/roteiro1/pre/encoding.py"
    ```

#### Normalização e Padronização
 
Depois da limpeza e codificação, aplicamos técnicas de **reescalonamento dos dados numéricos**:  

- **Min-Max Normalization**: transforma os valores de cada coluna para o intervalo `[0, 1]`. Essa técnica é útil quando queremos preservar a proporção entre valores, mas garantir que todos estejam no mesmo intervalo.  
- **Standardization (Z-Score)**: transforma os dados para que tenham **média 0 e desvio padrão 1**, centralizando a distribuição. Essa técnica é mais adequada quando os dados seguem (ou se aproximam de) uma distribuição normal.  

Ambas as técnicas têm como objetivo **evitar que variáveis com escalas diferentes dominem o aprendizado** dos algoritmos de Machine Learning.

=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/roteiro1/pre/norm_stand.py"
    ```

=== "Original"

    ```python exec="on" html="0"
    --8<-- "docs/roteiro1/pre/norm_minmax.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/roteiro1/pre/pre_processing.py"
    ```
---

### 3. Divisão dos Dados
Após o pré-processamento do dataset (preenchimento de valores ausentes, codificação de variáveis categóricas e normalização/padronização), os dados foram divididos em **conjuntos de treino e teste**.  
- **Treinamento (70%)**: utilizado para ajustar os parâmetros do modelo Decision Tree.  
- **Teste (30%)**: utilizado para avaliar o desempenho do modelo em dados não vistos.  

No código, a divisão foi feita com `train_test_split` do scikit-learn, garantindo uma separação aleatória, mas reproduzível com `random_state=42`.

=== "Code"

    ```python
    # dividir variáveis
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```
---

### 4. Treinamento do Modelo
O modelo **Decision Tree Classifier** foi criado e treinado com os dados de treino (`X_train` e `y_train`).  
- Cada nó da árvore representa uma decisão baseada em uma feature.  
- Cada folha da árvore representa a classe final prevista (`0 = No Disease` ou `1 = Disease`).  

O treinamento foi realizado com:
=== "Code"

    ```python
    classifier = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    classifier.fit(X_train, y_train)
    ```
---

### 5. Avaliação do Modelo
O modelo treinado foi avaliado utilizando o conjunto de teste `(X_test, y_test)`.

- **Acurácia (Accuracy)**: proporção de previsões corretas.
- **Visualização da árvore**: interpretação das regras de decisão aprendidas pelo modelo.

No código, a avaliação foi feita com:
=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/roteiro1/arvore/arvore.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/roteiro1/arvore/arvore.py"
    ```
---

### 6. Relatório Final
O desenvolvimento do projeto começou pela exploração do conjunto de dados, onde foram observadas características importantes como a predominância do sexo masculino e a maior probabilidade de diagnóstico positivo entre pacientes assintomáticos. Também foi identificada a presença de valores extremos em colesterol e pressão arterial, o que reforçou a necessidade de um bom pré-processamento.

O resultado obtido apresentou uma **acurácia em torno de 83%**, mostrando que o modelo conseguiu capturar bem os padrões presentes no conjunto de dados. A análise da árvore evidenciou a relevância de variáveis como `ChestPainType`, `ST_Slope` e `Oldpeak`.

Embora os resultados tenham sido positivos, algumas melhorias podem ser consideradas. Entre elas, destacar o uso de mais de um modelo para comparar desempenhos, a aplicação de validação cruzada para obter uma medida mais estável e a análise de métricas além da acurácia, como precisão e recall. Essas ações simples já poderiam aumentar a confiabilidade do modelo e oferecer uma visão mais completa de seu desempenho.

Em resumo, o projeto cumpriu o objetivo de construir um protótipo capaz de prever a presença de doenças cardiovasculares a partir de variáveis clínicas, apresentando bons resultados e mostrando potencial para evoluir em versões futuras.