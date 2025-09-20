import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score




df = pd.read_csv('https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/base/Obesity%20Classification.csv')

#EUA IA é a tabela que se refere ao id do individuo
df = df.drop(columns=['EUA IA'])

label_encoder = LabelEncoder()  
df['Label'] = label_encoder.fit_transform(df['Label'])

label_encoder = LabelEncoder()  
df['Gender'] = label_encoder.fit_transform(df['Gender'])

x = df.drop(columns=['Label'])
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(df.to_markdown(index=False))