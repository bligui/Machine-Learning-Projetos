## Projeto - Obesity Classification
---
Esse projeto tem como objetivo aplicar técnicas de **Machine Learning**, abordando Árvore de Decisão, KNN e K-Means, para prever a presença de obesidade a partir de variáveis físicas.

---

- Fonte: [Obesity Classification](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset)

- Formato: 108 observações x 7 colunas.

- Colunas: 
    - ID: Identificador
    - Age: Idade do indivíduo
    - Gender: Sexo (Male/Female)
    - Height: Altura (Em CM)
    - Weight: Peso (Em KG)
    - BMI: Em português, IMC. Índice de massa corporal.
    - Label: Variável alvo, com 4 classes:
        - `Underweight` (0)
        - `Normal Weight` (1)
        - `Overweight` (2)
        - `Obese` (3)

*Observação: A coluna `ID` foi removida das features, identificador não informativo.*

---

### Exploração de Dados

- Distribuição de classes:

| Classe | Contagem |
|---|---|
| Underweight | 47 |
| Normal Weight | 29 |
| Overweight | 20 |
| Obese | 12 |

Existe desbalanceamento pronunciado (Underweight ≈ 4× Obese). 
Isso exige cuidado: usar estratificação no split, aplicar técnicas de oversampling (SMOTE) apenas no conjunto de treino ou utilizar `class_weight` em modelos que aceitam.

#### Verificação de `BMI`
Ao recalcular BMI a partir de Height e Weight (assumindo Height em cm, convertido para metros), encontramos inconsistência significativa entre a coluna BMI fornecida e o BMI calculado:

- 105 de 108 linhas (≈ 97.22%) têm diferença absoluta |BMI - BMI_calc| > 0.2.

Confirmar a origem e unidade das colunas; recalcular BMI a partir de Height/Weight e usar o BMI recalculado (ou remover a coluna BMI original), pois valores inconsistentes podem introduzir ruído severo.

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/projeto1/Exploracaodedados.py"
    ```

### Pré-Processamento

- Remoção de `ID`
- Mapeamento da target `Label` para inteiros:
    - Underweight: 0
    - Normal: 1
    - Overweight: 2
    - Obese: 3
- `Gender` codificado com `LabelEncoder` (0/1).
- Conversão de variáveis categóricas com pd.get_dummies(..., drop_first=True) para modelos que exigem entradas numéricas.
- Para KMeans: aplicação de `StandardScaler` antes do PCA/clusterização.

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/projeto1/Preprocessamento.py"
    ```

### Divisão de dados
Separar 70% para treino e 30% para teste: 

`train_test_split(test_size=0.3, random_state=42, stratify=y).`

**Treino (70%)**

| Label (num) | Contagem |
|---|---|
| 0 (Underweight) | 33 |
| 1 (Normal Weight) | 20 |
| 2 (Overweight) | 14 |
| 3 (Obese) | 8 |

**Teste (30%)**

| Label (num) | Contagem |
|---|---|
| 0 (Underweight) | 14 |
| 1 (Normal Weight) | 9 |
| 2 (Overweight) | 6 |
| 3 (Obese) | 4 |

### Modelagem e resultados

Configuração geral: `random_state = 42`em operações determinísticas; preservamos estratificação no split para todas as avaliações.

---

### Decision Tree

- Resultado: **Accuracy = 0.9697 (≈ 96.97%).**


=== "Decision Tree"

    ```python exec="1" html="true"
    --8<-- "docs/projeto1/Avaliacaodomodelo.py"
    ```
=== "Code"

    ```python exec="0"
    --8<-- "docs/projeto1/Avaliacaodomodelo.py"
    ```
---
Classification report (precision / recall / f1 / support):

- Underweight: 
    - precision=1.00, 
    - recall=1.00, 
    - f1=1.00 (support=14)

- Normal: 
    - precision=0.90, 
    - recall=1.00, 
    - f1≈0.947 (support=9)

- Overweight: 
    - precision=1.00, 
    - recall≈0.833, 
    - f1≈0.909 (support=6)

- Obese: 
    - precision=1.00,
    - recall=1.00,
    - f1=1.00 (support=4)

Interpretação: desempenho muito alto no conjunto de teste, atenção ao overfitting, especialmente com árvores não podadas em datasets pequenos. Recomenda-se validação com CV e ajuste de hiperparâmetros (max_depth, min_samples_leaf, ccp_alpha).

---

### KNN

Resultado: **Accuracy = 0.9394 (≈ 93.94%).**
5‑fold CV: média ≈ 0.879654, desvio ≈ 0.022436.

=== "KNN"

    ```python exec="1" html="1"
    --8<-- "docs/projeto1/TreinamentoKNN.py"
    ```
=== "Code"

    ```python exec="0"
    --8<-- "docs/projeto1/TreinamentoKNN.py"
    ```

Interpretação: KNN com Height & Weight separa bem as classes; performance sensível a k e escalonamento. Recomenda-se testar StandardScaler e GridSearchCV para n_neighbors e weights.

---

### KMeans
PCA (2 componentes):

- PC1: 0.5597427 (≈55.97% da variância)

- PC2: 0.3135229 (≈31.35% da variância)

- Soma ≈ 0.873266 (≈87.33%)

Inertia (KMeans):

- k = 4 → inertia = 96.192596

- k = 5 → inertia = 68.679993

=== "kmeans"

    ```python exec="1" html="1"
    --8<-- "docs/projeto1/kmeans.py"
    ```
=== "code"

    ```python exec="0"
    --8<-- "docs/projeto1/kmeans.py"
    ```

Interpretação: os dois primeiros PCs explicam ~87% da variância, logo a projeção 2D é representativa. k=5 apresenta menor inertia, mas a escolha final de k deve considerar silhouette score e interpretação dos clusters em relação às classes reais.

---

### Conclusão
- Modelos testados (Decision Tree e KNN) apresentam desempenho elevado no conjunto de teste (≈97% e ≈94%, respectivamente). Entretanto, o dataset é pequeno e possui desbalanceamento e inconsistência crítica na coluna BMI.
