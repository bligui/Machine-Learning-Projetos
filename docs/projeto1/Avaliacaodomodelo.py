import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score




df = pd.read_csv('https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/base/Obesity%20Classification.csv')

#Eé a tabela que se refere ao id do individuo
df = df.drop(columns=['ID'])


df["Label"] = df["Label"].map({
    "Underweight": 0,
    "Normal Weight": 1,
    "Overweight": 2,
    "Obese": 3
})

label_encoder = LabelEncoder()  
df['Gender'] = label_encoder.fit_transform(df['Gender'])

x = df.drop(columns=['Label'])
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# para Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier(random_state=42)
classifier.fit(x_train, y_train)

plt.figure(figsize=(12,10))

# Avaliação o modelo, medindo a acuracia
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)


buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())



