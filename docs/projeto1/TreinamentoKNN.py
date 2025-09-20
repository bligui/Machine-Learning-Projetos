import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/base/Obesity%20Classification.csv')

#Eé a tabela que se refere ao id do individuo
df = df.drop(columns=['ID'])


df["Label"] = df["Label"].map({
    "Normal Weight": 0,
    "Obese": 1
})

label_encoder = LabelEncoder()  
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X = df[['Height', 'Weight']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Treianamento do KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


#Teste e validação
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")



labels_map = {0: "Normal Weight", 1: "Obese"}
y_labels = y.map(labels_map)


#Preparação para o gráfico da fronteira de decisão(malha de visualização)
h = 0.02
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

#Prevendo classe em cada ponto
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#gráfico final
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn_r, alpha=0.3)
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y_labels, style=y_labels, palette={'Normal Weight': 'green', 'Obese': 'red'}, s=100) #motivooooooo do errroo
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("KNN Decision Boundary (k=3) -  Diagnóstico de Obesidade")
plt.legend(title="Diagnóstico de obesidade")  



#Exibição do gráfico
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())





