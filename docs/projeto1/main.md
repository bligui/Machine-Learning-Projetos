# exploração de dados


1. Exploração de Dados (EDA)

Objetivo: entender os dados, verificar distribuições e balanceamento das classes.

O target (Label) tem 4 classes, precisamos ver se há desbalanceamento (muitas instâncias de uma classe e poucas de outra). Classes: Peso normal, sobrepeso, abaixo do peso e obeso.

Balanceamento das classes

Underweight: 47

Normal Weight: 29

Overweight: 20

Obese: 12

aplicar oversampling

Temos desbalanceamento: a classe Underweight tem quase 4x mais exemplos que Obese. Isso pode fazer a árvore “puxar” mais para as classes majoritárias.


Verificar se BMI está realmente consistente com Weight/Height, porque pode ser uma variável redundante.
verificando ele não é

dataset (pequeno, com poucas instâncias), o mais indicado é:

Usar estratificação na divisão treino/teste.

Aplicar SMOTE no treino.

=== "Code"

    ```python
    --8<-- "docs/projeto1/Exploracaodedados.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/projeto1/Exploracaodedados.py"
    ```


# Pré - Processamento



=== "Code"

    ```python
    --8<-- "docs/projeto1/Preprocessamento.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/projeto1/Preprocessamento.py"
    ```