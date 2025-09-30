import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/base/Obesity%20Classification.csv')

# Features (remover a target e transformar variáveis categóricas em dummies)
X = pd.get_dummies(df.drop(columns=['Label']), drop_first=True)

# Escalar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redução de dimensionalidade PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Adicionar clusters ao DataFrame original
df['Cluster'] = labels

# Visualização dos clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centróides')
plt.title('Clusters após redução de dimensionalidade (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# Tabela de variância explicada
variancias = pca.explained_variance_ratio_
tabela_variancia = pd.DataFrame({
    'Componente Principal': [f'PC{i+1}' for i in range(len(variancias))],
    'Variância Explicada': variancias,
    'Variância Acumulada': np.cumsum(variancias)
})

print(tabela_variancia.to_markdown(index=False))

# Variância total explicada
print("\nVariância total explicada (2 componentes):", np.sum(variancias))

# Resultados do KMeans
print("\nCentróides finais (no espaço PCA):", kmeans.cluster_centers_)
print("Inércia (WCSS):", kmeans.inertia_)
